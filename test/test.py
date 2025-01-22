from interval import interval, inf

class CustomInterval(interval):
    def __new__(cls, *args):
        # Handle the [a, b] syntax
        if len(args) == 1 and isinstance(args[0], slice):
            a, b = args[0].start, args[0].stop
            if cls._is_scalar(a) and cls._is_scalar(b):
                # Adjust the range by expanding it
                return super(CustomInterval, cls).__new__(cls, (a - 0.5, b + 0.5))
            raise ValueError("Both start and stop must be scalars.")
        
        # Handle other cases (default behavior)
        return super(CustomInterval, cls).__new__(cls, *args)
    
    @staticmethod
    def _is_scalar(x):
        """Check if x is a scalar (not an interval)."""
        try:
            float(x)  # Scalars should be convertible to float
            return True
        except (TypeError, ValueError):
            return False

if __name__ == "__main__":
    a = interval(CustomInterval[1, 2])
    b = interval(CustomInterval[3, 4])
    print(a | b)
