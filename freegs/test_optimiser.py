from . import optimiser

# Tests and example of solving optimisation problem
#
# Objects being evolved are lists of coefficients,
# and the optimiser is minimising a polynomial (a quadratic).
#
# Run this script to show animation of the solution, or run
# tests using pytest.


class ControlIndex:
    """
    Modify an object (e.g. list of coefficients) using an index
    """

    def __init__(self, index):
        self.index = index

    def set(self, coefs, value):
        coefs[self.index] = value

    def get(self, coefs):
        return coefs[self.index]


def calculate_score(coefs):
    """
    Calculates (x-1)^2 + (y-2)^2
    so minimum is at x=1, y=2
    """
    return (coefs[0] - 1.0) ** 2 + (coefs[1] - 2.0) ** 2


def test_quadratic():
    # Check that a quadratic can be solved
    # Note that the optimiser uses random numbers so
    # there is a small chance of this test failing even if correct
    start_values = [0.1, 0.1]  # Starting guess

    # The optimiser can control two coefficients,
    # and should use calculate_score to evaluate solutions
    result = optimiser.optimise(
        start_values, [ControlIndex(0), ControlIndex(1)], calculate_score, maxgen=300
    )

    # Answer should be close to (1,2)
    assert abs(result[0] - 1.0) < 1e-2 and abs(result[1] - 2.0) < 1e-2


def test_reducing():
    # Test that the best score never goes up
    start_values = [0.1, 0.1]  # Starting guess

    best_score = calculate_score(start_values)
    monitor_called = 0

    def monitor(gen, best, pop):
        nonlocal best_score
        nonlocal monitor_called
        score = best[0]
        assert score <= best_score
        best_score = score
        monitor_called += 1

    # Call monitor at each generation
    result = optimiser.optimise(
        start_values,
        [ControlIndex(0), ControlIndex(1)],
        calculate_score,
        maxgen=10,
        monitor=monitor,
    )
    assert monitor_called > 0
    assert result[0] <= best_score


### Internal components of optimiser algorithm


def test_pick_all():
    vals = optimiser.pickUnique(10, 10, [])
    # Should be a scrambled list of [0...10]

    assert vals != list(range(0, 10))  # Probably not sorted
    assert sorted(vals) == list(range(0, 10))


def test_pick_different():
    v1 = optimiser.pickUnique(20, 3, [])
    v2 = optimiser.pickUnique(20, 3, [])
    assert v1 != v2


def test_pick_excludes():
    for i in range(20):
        vals = optimiser.pickUnique(10, 3, [1, 8])
        assert 1 not in vals
        assert 8 not in vals


### Example optimiser with animation

if __name__ == "__main__":
    start_values = [0.1, 0.1]  # Starting guess

    import matplotlib.pyplot as plt

    fig = plt.figure()
    axis = fig.add_subplot(111)

    # Plot the exact solution
    axis.scatter([1.0], [2.0], c="blue", s=0.1)
    axis.axvline(1.0, ls="--")
    axis.axhline(2.0, ls="--")

    # A PathCollection representing the currently best point
    best_point = axis.scatter([start_values[0]], [start_values[1]], c="red")

    # Points representing the population
    pop_points = None

    def monitor(generation, best, pop):
        global best_point
        global pop_points
        global axis

        # Change the color of the previous points
        # to grey and light red
        best_point.set_color([1.0, 0.7, 0.7])
        if pop_points:
            pop_points.set_color([0.7, 0.7, 0.7])

        # Get all population values
        xs = [agent[1][0] for agent in pop]
        ys = [agent[1][1] for agent in pop]

        # Plot new points in black and red
        pop_points = axis.scatter(xs, ys, c="black")

        best_point = axis.scatter([best[1][0]], [best[1][1]], c="red")

        axis.figure.canvas.draw()
        axis.set_title("Generation: {}, Best score: {}".format(generation, best[0]))
        plt.pause(0.5)

    result = optimiser.optimise(
        start_values,
        [ControlIndex(0), ControlIndex(1)],
        calculate_score,
        maxgen=100,
        monitor=monitor,
    )
    plt.show()
