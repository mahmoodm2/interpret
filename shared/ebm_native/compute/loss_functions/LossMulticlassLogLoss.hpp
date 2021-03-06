// Copyright (c) 2018 Microsoft Corporation
// Licensed under the MIT license.
// Author: Paul Koch <code@koch.ninja>

// !! To add a new loss/objective function in C++ follow the steps at the top of the "loss_registrations.hpp" file !!

#include "Loss.hpp"

// TFloat could be double, float, or some SIMD intrinsic type
template<typename TFloat>
struct LossMulticlassLogLoss : public LossMulticlass {
   LOSS_CLASS_BOILERPLATE(LossMulticlassLogLoss, true, 1)

   size_t m_countTargetClasses;

   // IMPORTANT: the constructor parameters here must match the RegisterLoss parameters in the file Loss.cpp
   INLINE_ALWAYS LossMulticlassLogLoss(const Config & config) {
      UNUSED(config);

      if(1 == config.cOutputs) {
         // we share the tag "log_loss" with binary classification
         throw SkipRegistrationException();
      }

      if(config.cOutputs <= 0) {
         throw ParameterMismatchWithConfigException();
      }

      m_countTargetClasses = config.cOutputs;
   }

   GPU_DEVICE INLINE_ALWAYS TFloat CalculatePrediction(TFloat score) const {
      //TODO implement
      return -score * 999;
   }

   GPU_DEVICE INLINE_ALWAYS TFloat CalculateGradient(TFloat target, TFloat prediction) const {
      //TODO implement
      return 999.9999;
   }

   // if the loss function doesn't have a second derivative, then delete the CalculateHessian function.
   GPU_DEVICE INLINE_ALWAYS TFloat CalculateHessian(TFloat target, TFloat prediction) const {
      //TODO implement
      return 999.9999;
   }
};
