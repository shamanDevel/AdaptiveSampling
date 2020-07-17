#pragma once

#include "commons.h"

#ifdef RENDERER_HAS_RENDERER
BEGIN_RENDERER_NAMESPACE

/**
 * \brief Given a function f, piecewise linear defined by 'heights',
 * evaluate it at the new sample positions 'newX'.
 *
 * The sampling points 'heights' are of shape (Batch, N)
 * where they define the value of the function at regular intervals in
 * [0,1] (inclusive).
 * I.e. f(0) = heights[:, 0], f(1) = heights[:, N-1]
 * with equally spaced intervals in between.
 *
 * The tensor 'newX' of shape (Batch, M) specifies the M positions
 * where the function is evaluated.
 *
 * \param heights the sample point heights of the function (Batch, N)
 * \param newX the locations where the function is evaluated (Batch, M)
 * \param output the values at those evaluated locations (Batch, M)
 */
MY_API void RegularInterp1d(torch::Tensor heights, torch::Tensor newX, torch::Tensor output);

END_RENDERER_NAMESPACE
#endif
