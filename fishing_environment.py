import math

class FishingEnvironment:
    """
    Manages the shared fish resource in the simulation.
    """
    def __init__(self, initial_fish: int, regeneration_rate: float):
        """
        Initializes the fishing environment.

        Args:
            initial_fish: The starting number of fish.
            regeneration_rate: The factor by which the fish population multiplies each round.
        """
        if initial_fish < 0:
            raise ValueError("Initial fish count cannot be negative.")
        if regeneration_rate <= 0:
            raise ValueError("Regeneration rate must be positive.")

        self.current_fish = float(initial_fish) # Store as float for regeneration calculations
        self.regeneration_rate = regeneration_rate
        self.initial_fish = initial_fish

    def get_fish_count(self) -> int:
        """Returns the current number of fish, rounded down."""
        return math.floor(self.current_fish)

    def take_fish(self, amount: int) -> int:
        """
        Removes a specified amount of fish from the environment.

        Args:
            amount: The non-negative number of fish agents attempt to take.

        Returns:
            The actual number of fish taken, potentially less than requested.
        """
        if amount < 0:
            # Avoid allowing negative takes, although callers should ideally prevent this.
            print("Warning: Attempted to take a negative amount of fish. Taking 0.")
            return 0

        available = self.get_fish_count()
        taken_fish = min(amount, available)
        self.current_fish -= taken_fish
        self.current_fish = max(0.0, self.current_fish) # Ensure float stays non-negative
        return taken_fish

    def regenerate(self):
        """
        Applies the regeneration rate to the current fish population.
        """
        self.current_fish *= self.regeneration_rate

    def get_state(self) -> dict:
        """Returns the current state of the environment."""
        return {
            "current_fish": self.get_fish_count(),
            "regeneration_rate": self.regeneration_rate
        }

    def __str__(self) -> str:
        return f"FishingEnvironment(Fish: {self.get_fish_count()}, Regen Rate: {self.regeneration_rate})" 