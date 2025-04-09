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
            regeneration_rate: The factor by which the fish population multiplies each round (e.g., 1.1 for 10% growth).
        """
        if initial_fish < 0:
            raise ValueError("Initial fish count cannot be negative.")
        if regeneration_rate <= 0:
            raise ValueError("Regeneration rate must be positive.")

        self.current_fish = initial_fish
        self.regeneration_rate = regeneration_rate
        self.initial_fish = initial_fish # Keep track for potential resets or analysis

    def get_fish_count(self) -> int:
        """Returns the current number of fish (rounded down)."""
        return math.floor(self.current_fish)

    def take_fish(self, amount: int) -> int:
        """
        Attempts to remove a specified amount of fish from the environment.

        Args:
            amount: The number of fish agents attempt to take.

        Returns:
            The actual number of fish taken, which may be less than requested if
            the stock is insufficient. Returns 0 if the amount is negative.
        """
        if amount < 0:
            print("Warning: Attempted to take a negative amount of fish. Taking 0.")
            return 0

        taken_fish = min(amount, self.get_fish_count())
        self.current_fish -= taken_fish
        # Ensure fish count doesn't go below zero due to floating point issues after regeneration
        self.current_fish = max(0, self.current_fish)
        return taken_fish

    def regenerate(self):
        """
        Applies the regeneration rate to the current fish population.
        The population grows based on the rate, but is capped implicitly
        by the fact that fish are taken before regeneration in a round.
        The result is stored as a float but accessed as int via get_fish_count.
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