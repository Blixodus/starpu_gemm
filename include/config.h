#ifndef _config_h_
#define _config_h_

#cmakedefine ENABLE_MPI
#cmakedefine ENABLE_STARPU
#cmakedefine ENABLE_CUDA

#if defined(ENABLE_MPI)
#include <mpi.h>
#endif

#if defined(ENABLE_STARPU)
#include <starpu.h>
#if defined(ENABLE_MPI)
#include <starpu_mpi.h>
#endif
#endif

#endif /* _config_h_ */
