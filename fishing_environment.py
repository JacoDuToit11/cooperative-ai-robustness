import math

class ResourceEnvironment:
    """
    Manages a shared resource that doubles each round, capped at a limit.
    """
    def __init__(self, initial_resource: float, resource_limit: float):
        """
        Initializes the environment.

        Args:
            initial_resource: The starting amount of the resource.
            resource_limit: The maximum resource level the environment can sustain.
        """
        if initial_resource < 0:
            raise ValueError("Initial resource cannot be negative.")
        if resource_limit <= 0:
            raise ValueError("Resource limit must be positive.")

        self.resource_limit = resource_limit
        # Ensure initial resource does not exceed limit
        self.current_resource = min(float(initial_resource), self.resource_limit)
        self.initial_resource = initial_resource

    def get_resource_level(self) -> float:
        """Returns the current resource level."""
        return self.current_resource

    def take_resource(self, amount: float) -> float:
        """
        Removes a specified amount of resource from the environment.

        Args:
            amount: The non-negative amount agents attempt to take.

        Returns:
            The actual amount taken, potentially less than requested.
        """
        if amount < 0:
            print("Warning: Attempted to take a negative amount. Taking 0.")
            return 0.0

        available = self.get_resource_level()
        taken_resource = min(amount, available)
        self.current_resource -= taken_resource
        self.current_resource = max(0.0, self.current_resource)
        return taken_resource

    def regenerate(self):
        """
        Doubles the current resource level, capped by the resource limit.
        """
        if self.current_resource <= 0:
            return # No growth if resource is depleted

        self.current_resource *= 2
        # Ensure resource does not exceed limit after doubling
        self.current_resource = min(self.current_resource, self.resource_limit)

    def get_state(self) -> dict:
        """Returns the current state of the environment."""
        return {
            "current_resource": self.get_resource_level(),
            "resource_limit": self.resource_limit
        }

    def __str__(self) -> str:
        return f"ResourceEnvironment(Resource: {self.current_resource:.2f}, Limit: {self.resource_limit:.2f})" 