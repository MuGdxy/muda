#pragma once
#include "check.h"
#define checkCudaErrors(val) ::muda::check((val), #val, __FILE__, __LINE__)