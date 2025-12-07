import itertools
import logging

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging: force root logger to have a stream handler so output is visible
logging.basicConfig(level=logging.WARN, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
NUM_COUNTERS = 1  # Number of ID counters at the checkpoint
SIM_TIME = 20  # Simulation time in minutes
NUM_SCANNERS = 1
SCANNER_LOW = 0.5
SCANNER_HIGH = 1.0
BETA1 = 0.2
BETA2 = 0.75


# # Function to run the simulation and return a DataFrame
def run_simulation(
    num_counters=NUM_COUNTERS,
    num_scanners=NUM_SCANNERS,
    sim_time=SIM_TIME,
    random_seed=RANDOM_SEED,
    beta1=BETA1,
    beta2=BETA2,
    initial_passengers=2,
    log_level="WARN",
):
    rng = np.random.default_rng(random_seed)
    data = []
    logger.setLevel(log_level)

    def NoInSystem(R: simpy.Resource):
        """Total number of customers in the resource R"""
        # Support resources that may not have put_queue/users attributes
        return max(0, len(getattr(R, "put_queue", [])) + len(getattr(R, "users", [])))

    class Scanners:
        """Passenger scanners helper.

        Maintains a list of scanner resources and exposes a Customer process
        that a arriving passenger can use: choose shortest queue, wait, then
        take a uniform scanning time between wait_low and wait_high.
        """

        def __init__(self, env, num_scanners, wait_low, wait_high):
            self.env = env
            self.num_scanners = num_scanners
            self.wait_low = wait_low
            self.wait_high = wait_high
            self.scanners = [simpy.Resource(env) for _ in range(self.num_scanners)]

        def queue_lengths(self):
            return list(map(NoInSystem, self.scanners))

        def queue(
            self,
            name,
            generation_time,
            id_check_wait,
            abs_time_reach_id,
            time_pass_id,
            abs_time_exit_id,
        ):
            """Process for a passenger to join a scanner queue and be scanned.

            Uses the scanner list and the class's environment so multiple Customer
            processes (one per passenger) run in parallel and form queues per scanner.
            """
            arrive = self.env.now
            Qlength = self.queue_lengths()

            logger.debug(
                f"t={self.env.now:7.4f} | {name} | Arrives at scanner | Queue lengths: {Qlength}"
            )
            # choose first scanner with minimal queue length
            choice = min(range(len(Qlength)), key=lambda i: Qlength[i])
            with self.scanners[choice].request() as req:
                yield req
                wait = self.env.now - arrive
                time_before_scan = self.env.now
                logger.debug(
                    f"t={self.env.now:7.4f} | {name} | Begins scanning at scanner {choice} | Wait time: {wait:6.4f}"
                )
                time_in_line = rng.uniform(self.wait_low, self.wait_high)
                yield self.env.timeout(time_in_line)
                total_time = self.env.now - generation_time
                abs_time_leave_scanner = self.env.now
                logger.debug(
                    f"t={self.env.now:7.4f} | {name} | Completes scanning at scanner {choice} | Scan duration: {time_in_line:6.4f}"
                )
                logger.info(
                    f"t={self.env.now:7.4f} | {name} | Exits system | Total time in process: {total_time:6.4f}"
                )
                data.append(
                    {
                        "name": name,
                        "generation_time": generation_time,
                        "id_check_wait": id_check_wait,
                        "scanner_queues": Qlength,
                        "scan_line": choice,
                        "scan_wait": wait,
                        "total_time": total_time,
                        "abs_time_reach_id": abs_time_reach_id,
                        "time_pass_id": time_pass_id,
                        "abs_time_exit_id": abs_time_exit_id,
                        "abs_time_reach_scanner": arrive,
                        "abs_time_leave_scanner": abs_time_leave_scanner,
                    }
                )

    class Checkpoint:
        """A security checkpoint has a limited number of ID counters
        to check passengers in parallel.

        Passengers have to request one of the counters. When they get one,
        they can start the ID checking process and wait for it to finish (which
        takes ``check_time`` minutes).
        """

        def __init__(self, env, num_counters, beta2):
            self.env = env
            self.counter = simpy.Resource(env, num_counters)
            self.beta2 = beta2

        def check_id(self, passenger):
            """The ID checking process. It takes a ``passenger`` process
            and simulates checking their ID."""
            rate = rng.exponential(self.beta2)
            logger.debug(
                f"t={self.env.now:7.4f} | {passenger} | ID check duration: {rate:6.4f}"
            )
            yield self.env.timeout(rate)

    def passenger(env, name, cp, scanners, generation_time):
        """The passenger process (each passenger has a ``name``) arrives
        at the checkpoint (``cp``) and requests an ID counter. After ID check
        they may go to scanners if provided.
        """
        arrive_time = env.now
        logger.debug(
            f"t={env.now:7.4f} | {name} | Arrives at checkpoint | Queue length: {NoInSystem(cp.counter)}"
        )

        with cp.counter.request() as request:
            yield request
            id_check_wait = env.now - arrive_time
            time_before_id_check = env.now
            logger.debug(
                f"t={env.now:7.4f} | {name} | Begins ID check | Wait time: {id_check_wait:6.4f}"
            )
            yield env.process(cp.check_id(name))
            abs_time_exit_id = env.now
            time_pass_id = abs_time_exit_id - time_before_id_check
            logger.debug(f"t={env.now:7.4f} | {name} | Completes ID check")

        env.process(
            scanners.queue(
                name,
                generation_time,
                id_check_wait,
                arrive_time,
                time_pass_id,
                abs_time_exit_id,
            )
        )

    def setup(
        env,
        num_counters,
        beta1,
        beta2,
        num_scanners,
        scan_low,
        scan_high,
        initial_passengers,
    ):
        """Create a passenger arrival process at the checkpoint."""
        # Create the checkpoint
        checkpoint = Checkpoint(env, num_counters, beta2)
        scanners = Scanners(env, num_scanners, scan_low, scan_high)

        passenger_count = itertools.count()

        # Create n initial passengers
        for _ in range(initial_passengers):
            passenger_id = next(passenger_count)
            logger.debug(
                f"t={env.now:7.4f} | Passenger {passenger_id} | Generated at start"
            )
            env.process(
                passenger(
                    env, f"Passenger {passenger_id}", checkpoint, scanners, env.now
                )
            )

        # Create more passengers while the simulation is running
        while True:
            arrival_delay = rng.exponential(beta1)
            yield env.timeout(arrival_delay)
            passenger_id = next(passenger_count)
            logger.debug(
                f"t={env.now:7.4f} | Passenger {passenger_id} | Generated after {arrival_delay:6.4f} delay"
            )
            env.process(
                passenger(
                    env, f"Passenger {passenger_id}", checkpoint, scanners, env.now
                )
            )

    env = simpy.Environment()
    env.process(
        setup(
            env,
            num_counters=num_counters,
            beta1=beta1,
            beta2=beta2,
            num_scanners=num_scanners,
            scan_low=SCANNER_LOW,
            scan_high=SCANNER_HIGH,
            initial_passengers=initial_passengers,
        )
    )

    env.run(until=sim_time)

    return pd.DataFrame(data)
