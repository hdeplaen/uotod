# import scipy

def hungarian(C):
    r"""
    Performs the Hungarian matching in a sequential way (on CPU).

    :param C: Cost matrix associated to the match.
    :type C: Tensor[num, n_pred, n_gt]
    :return: Hungarian match
    :rtype: Tensor[num, n_pred, n_gt]
    """

    return None