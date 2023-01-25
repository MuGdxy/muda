# Implement

## launch

| launch       | details                                                |
| ------------ | ------------------------------------------------------ |
| launch       | basic cuda launch wrapper                              |
| parallel_for | dynamic grid dim & grid stride loop based parallel for |

## viewer

| viewer              | full name                  | details                                         |
| ------------------- | -------------------------- | ----------------------------------------------- |
| mapper1D/2D/3D      | mapper                     | base class for idxerND                          |
| idxer/idxer1D/2D/3D | indexer N Dimension        | array-like dense                                |
| cse                 | compressed sparse elements | sparse 2d array: cse(i,j) -> data[begin[i] + j] |
| csr                 | compressed sparse row      | sparse matrix  CSR format                       |

## graph

| graph        | details                                                      |
| ------------ | ------------------------------------------------------------ |
| graph        | basic wrapper of cuda graph, get node parms from `<launch>.asNodeParms()` |
| graphManager | a way to automatically setup dependencies between nodes by checking the resource usage of kernels, core api `<launch>.addNode`. |
|              |                                                              |

## buffer

| buffer        | details                         |
| ------------- | ------------------------------- |
| device_buffer | async device_vector-like buffer |
|               |                                 |
|               |                                 |

## container

| container                    | details |
| ---------------------------- | ------- |
| host/universal/device_vector |         |
| host/universal/device_var    |         |

## algorithm

| algorithm             | sub-algorithm                |
| --------------------- | ---------------------------- |
| DeviceRunLengthEncode | Encode                       |
| DeviceRadixSort       | SortPairs/SortKeys           |
| DeviceScan            | InclusiveScan/ExclusiveScan/ |
| DeviceReduce          | Reduce/Max                   |

## thread_only

| thread_only    | details |
| -------------- | ------- |
| vector         |         |
| priority_queue |         |
|                |         |

## blas

## sparse

## solver

