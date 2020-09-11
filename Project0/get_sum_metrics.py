def get_sum_metrics(predictions, metrics=None):

    if metrics is None:
        metrics = []

    for i in range(3):
        metrics.append((lambda y: lambda x: x + y)(i))

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return(sum_metrics)