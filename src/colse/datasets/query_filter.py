


from loguru import logger
import numpy as np
import tqdm


def filter_queries(query_l, query_r, col_ids_to_remove):
    """
    Filter queries based on the columns to remove
    """

    # remove the columns from the queries
    query_l = np.delete(query_l, col_ids_to_remove, axis=1)
    query_r = np.delete(query_r, col_ids_to_remove, axis=1)
    query_length = query_l.shape[0]
    """remove all -inf and +inf values"""
    remove_indexes = []
    loop = tqdm.tqdm(range(query_length), desc="Filtering queries")
    for i in loop:
        """check is inf length is equal to the query length"""
        lb = query_l[i, :]
        ub = query_r[i, :]
        if np.all(lb == -np.inf) and np.all(ub == np.inf):
            remove_indexes.append(i)

        loop.set_postfix({"removed": len(remove_indexes)})
    
    if len(remove_indexes) > 0:
        logger.info(f"Removing {len(remove_indexes)} queries with all -inf and +inf values")
        query_l = np.delete(query_l, remove_indexes, axis=0)
        query_r = np.delete(query_r, remove_indexes, axis=0)

    new_query_length = query_l.shape[0]
    logger.info(f"Query length reduced from {query_length} to {new_query_length}")

    return query_l, query_r
