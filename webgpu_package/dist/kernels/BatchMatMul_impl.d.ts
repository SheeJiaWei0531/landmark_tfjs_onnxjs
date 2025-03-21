/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-backend-webgpu/dist/kernels/BatchMatMul_impl" />
import { backend_util, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '../backend_webgpu';
type BatchMatMulConfig = {
    a: TensorInfo;
    b: TensorInfo;
    transposeA: boolean;
    transposeB: boolean;
    backend: WebGPUBackend;
    bias?: TensorInfo;
    preluActivationWeights?: TensorInfo;
    leakyreluAlpha?: number;
    activation?: backend_util.Activation;
};
export declare function batchMatMulImpl({ a, b, transposeA, transposeB, backend, bias, preluActivationWeights, leakyreluAlpha, activation }: BatchMatMulConfig): TensorInfo;
export {};
