def calculate_custom_accuracy(y_true, y_pred):
    """
    Calculate custom accuracy that considers the distance between categories.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Custom accuracy score.
    """
    category_distances = {
        (1, 2): 1,  # Distance from category 1 to 2
        (1, 3): 1.5,  # Distance from category 1 to 3
        (1, 4): 2  # Distance from category 1 to 4 (farthest)
    }

    total_distance = 0
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            distance = category_distances.get((true, pred), 0)
            total_distance += distance

    custom_accuracy = 1 - (total_distance / len(y_true))
    return custom_accuracy
