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
import { broadcast_util, env, util } from '@tensorflow/tfjs-core';
import { MatMulPackedProgram } from '../matmul_packed_webgpu';
import { MatMulReduceProgram } from '../matmul_reduce_webgpu';
import { MatMulSmallOutputSizeProgram } from '../matmul_small_output_size_webgpu';
import { BiasActivationProgram, MatMulSplitKProgram } from '../matmul_splitK_webgpu';
import { MatMulProgramType } from '../webgpu_util';
import { fill } from './Fill';
import { reshape } from './Reshape';
export function batchMatMulImpl({ a, b, transposeA, transposeB, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
    const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];
    const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
    const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];
    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);
    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);
    const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(a.shape.slice(0, -2), b.shape.slice(0, -2));
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);
    util.assert(innerShapeA === innerShapeB, () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
        `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
        `${b.shape} and transposeA=${transposeA}` +
        ` and transposeB=${transposeB} must match.`);
    const a3dShape = transposeA ?
        [batchDimA, innerShapeA, outerShapeA] :
        [batchDimA, outerShapeA, innerShapeA];
    const b3dShape = transposeB ?
        [batchDimB, outerShapeB, innerShapeB] :
        [batchDimB, innerShapeB, outerShapeB];
    // The rest of the implementation is designed to operate on rank-3 tensors
    const a3d = reshape({ inputs: { x: a }, backend, attrs: { shape: a3dShape } });
    const b3d = reshape({ inputs: { x: b }, backend, attrs: { shape: b3dShape } });
    const intermediates = [a3d, b3d];
    const batchDim = Math.max(batchDimA, batchDimB);
    const inputs = [a3d, b3d];
    const dimensions = [
        { type: 'int32', data: [outerShapeA] }, { type: 'int32', data: [outerShapeB] },
        { type: 'int32', data: [innerShapeA] }
    ];
    let program;
    let out;
    const outputShape = [batchDim, outerShapeA, outerShapeB];
    let matmulProgramType = env().get('WEBGPU_MATMUL_PROGRAM_TYPE');
    if (matmulProgramType < 0) {
        // Usually increasing workgroups is a good way to gain more performance for
        // few workgroups by tiling 32x32 (default matmul algorithm). Currently,
        // there are three ways to increase workgroups. 1) MatMulReduceProgram,
        // which is used only when the output size is very small (128 for now). 2)
        // MatMulSplitKProgram, increasing workgroups by spliting K. 3)
        // MatMulSmallOutputSizeProgram, increasing workgroups by small tile size.
        // For different devices, the minimum optimal workgroups may be different.
        // So here we set a |thresholdToIncreaseWorkgroups| to indicate whether we
        // need to increase workgroups. And the literal number is an empirical
        // value.
        const thresholdFlagValue = env().getNumber('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL');
        const thresholdToIncreaseWorkgroups = thresholdFlagValue > 0 ?
            thresholdFlagValue :
            backend.thresholdToIncreaseWorkgroups;
        const workgroupsBy32x32 = batchDim * Math.ceil(outerShapeA / 32) * Math.ceil(outerShapeB / 32);
        const hasFewWorkgroups = workgroupsBy32x32 <= thresholdToIncreaseWorkgroups ||
            (outerShapeA <= 8 &&
                workgroupsBy32x32 <= thresholdToIncreaseWorkgroups * 2);
        if (hasFewWorkgroups) {
            if (batchDim * outerShapeA * outerShapeB <= 128) {
                matmulProgramType = MatMulProgramType.MatMulReduceProgram;
            }
            else if (batchDim === 1 && innerShapeB >= 2000) {
                matmulProgramType = MatMulProgramType.MatMulSplitKProgram;
            }
            else {
                matmulProgramType = MatMulProgramType.MatMulSmallOutputSizeProgram;
            }
        }
        else {
            matmulProgramType = MatMulProgramType.MatMulPackedProgram;
        }
    }
    switch (matmulProgramType) {
        case MatMulProgramType.MatMulReduceProgram:
            program = new MatMulReduceProgram(outputShape, transposeA, transposeB, bias, activation, preluActivationWeights);
            break;
        case MatMulProgramType.MatMulSplitKProgram: {
            // The output buffer must be initailzed to zero before using since we
            // use atomicAdd in MatMulSplitKProgram.
            out = fill({ backend, attrs: { shape: outputShape, value: 0, dtype: a.dtype } });
            program = new MatMulSplitKProgram(outputShape, innerShapeB, transposeA, transposeB);
            if (bias || activation) {
                out =
                    backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
                const biasActivationProgram = new BiasActivationProgram(out.shape, bias, activation, preluActivationWeights);
                let uniformData = null;
                const activationInputs = [out];
                if (bias) {
                    activationInputs.push(bias);
                }
                if (preluActivationWeights) {
                    activationInputs.push(preluActivationWeights);
                }
                if (activation === 'leakyrelu') {
                    uniformData = [{ type: 'float32', data: [leakyreluAlpha] }];
                    biasActivationProgram.uniforms += ' alpha : f32,';
                }
                const outActivated = backend.runWebGPUProgram(biasActivationProgram, activationInputs, out.dtype, uniformData);
                intermediates.push(out);
                const outReshaped = reshape({ inputs: { x: outActivated }, backend, attrs: { shape: outShape } });
                intermediates.push(outActivated);
                for (const i of intermediates) {
                    backend.disposeData(i.dataId);
                }
                return outReshaped;
            }
            break;
        }
        case MatMulProgramType.MatMulSmallOutputSizeProgram:
            program = new MatMulSmallOutputSizeProgram(a3dShape, b3dShape, outputShape, transposeA, transposeB, bias, activation, preluActivationWeights);
            break;
        case MatMulProgramType.MatMulPackedProgram:
            // Experiments show that sequential access is more friendly for Intel
            // GPUs.
            const sequentialAccessByThreads = backend.adapterInfo.isIntel();
            program = new MatMulPackedProgram(a3dShape, outputShape, transposeA, transposeB, bias, activation, preluActivationWeights, sequentialAccessByThreads);
            break;
        default:
            throw new Error(`Unsupported MatMulProgramType ${matmulProgramType}.`);
    }
    if (bias) {
        inputs.push(bias);
    }
    if (preluActivationWeights) {
        inputs.push(preluActivationWeights);
    }
    if (activation === 'leakyrelu') {
        dimensions.push({ type: 'float32', data: [leakyreluAlpha] });
        program.uniforms += ' alpha : f32,';
    }
    out = backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
    const outReshaped = reshape({ inputs: { x: out }, backend, attrs: { shape: outShape } });
    intermediates.push(out);
    for (const i of intermediates) {
        backend.disposeData(i.dataId);
    }
    return outReshaped;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQmF0Y2hNYXRNdWxfaW1wbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvQmF0Y2hNYXRNdWxfaW1wbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQWUsY0FBYyxFQUFFLEdBQUcsRUFBYyxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUcxRixPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUM1RCxPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNoRixPQUFPLEVBQUMscUJBQXFCLEVBQUUsbUJBQW1CLEVBQUMsTUFBTSx5QkFBeUIsQ0FBQztBQUVuRixPQUFPLEVBQUMsaUJBQWlCLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUVqRCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFjbEMsTUFBTSxVQUFVLGVBQWUsQ0FBQyxFQUM5QixDQUFDLEVBQ0QsQ0FBQyxFQUNELFVBQVUsRUFDVixVQUFVLEVBQ1YsT0FBTyxFQUNQLElBQUksR0FBRyxJQUFJLEVBQ1gsc0JBQXNCLEdBQUcsSUFBSSxFQUM3QixjQUFjLEdBQUcsQ0FBQyxFQUNsQixVQUFVLEdBQUcsSUFBSSxFQUNDO0lBQ2xCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBQzdCLE1BQU0sS0FBSyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBRTdCLE1BQU0sV0FBVyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ3pFLE1BQU0sV0FBVyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBRXpFLE1BQU0sV0FBVyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ3pFLE1BQU0sV0FBVyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBRXpFLE1BQU0sVUFBVSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3hDLE1BQU0sVUFBVSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXhDLE1BQU0sU0FBUyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsVUFBVSxDQUFDLENBQUM7SUFDakQsTUFBTSxTQUFTLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxVQUFVLENBQUMsQ0FBQztJQUVqRCxNQUFNLGlCQUFpQixHQUFHLGNBQWMsQ0FBQywwQkFBMEIsQ0FDL0QsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoRCxNQUFNLFFBQVEsR0FBRyxpQkFBaUIsQ0FBQyxNQUFNLENBQUMsQ0FBQyxXQUFXLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQztJQUV0RSxJQUFJLENBQUMsTUFBTSxDQUNQLFdBQVcsS0FBSyxXQUFXLEVBQzNCLEdBQUcsRUFBRSxDQUFDLGtDQUFrQyxXQUFXLFNBQVM7UUFDeEQsR0FBRyxXQUFXLDRCQUE0QixDQUFDLENBQUMsS0FBSyxPQUFPO1FBQ3hELEdBQUcsQ0FBQyxDQUFDLEtBQUssbUJBQW1CLFVBQVUsRUFBRTtRQUN6QyxtQkFBbUIsVUFBVSxjQUFjLENBQUMsQ0FBQztJQUVyRCxNQUFNLFFBQVEsR0FBNkIsVUFBVSxDQUFDLENBQUM7UUFDbkQsQ0FBQyxTQUFTLEVBQUUsV0FBVyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUM7UUFDdkMsQ0FBQyxTQUFTLEVBQUUsV0FBVyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQzFDLE1BQU0sUUFBUSxHQUE2QixVQUFVLENBQUMsQ0FBQztRQUNuRCxDQUFDLFNBQVMsRUFBRSxXQUFXLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQztRQUN2QyxDQUFDLFNBQVMsRUFBRSxXQUFXLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFMUMsMEVBQTBFO0lBQzFFLE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxDQUFDLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLFFBQVEsRUFBQyxFQUFDLENBQUMsQ0FBQztJQUN6RSxNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxRQUFRLEVBQUMsRUFBQyxDQUFDLENBQUM7SUFDekUsTUFBTSxhQUFhLEdBQWlCLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBRS9DLE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsU0FBUyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRWhELE1BQU0sTUFBTSxHQUFpQixDQUFDLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztJQUN4QyxNQUFNLFVBQVUsR0FBRztRQUNqQixFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsV0FBVyxDQUFDLEVBQUMsRUFBRSxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsV0FBVyxDQUFDLEVBQUM7UUFDMUUsRUFBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBRSxDQUFDLFdBQVcsQ0FBQyxFQUFDO0tBQ3JDLENBQUM7SUFFRixJQUFJLE9BQXNCLENBQUM7SUFDM0IsSUFBSSxHQUFlLENBQUM7SUFDcEIsTUFBTSxXQUFXLEdBQ2IsQ0FBQyxRQUFRLEVBQUUsV0FBVyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQ3pDLElBQUksaUJBQWlCLEdBQUcsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLDRCQUE0QixDQUFXLENBQUM7SUFDMUUsSUFBSSxpQkFBaUIsR0FBRyxDQUFDLEVBQUU7UUFDekIsMkVBQTJFO1FBQzNFLHdFQUF3RTtRQUN4RSx1RUFBdUU7UUFDdkUsMEVBQTBFO1FBQzFFLCtEQUErRDtRQUMvRCwwRUFBMEU7UUFDMUUsMEVBQTBFO1FBQzFFLDBFQUEwRTtRQUMxRSxzRUFBc0U7UUFDdEUsU0FBUztRQUNULE1BQU0sa0JBQWtCLEdBQ3BCLEdBQUcsRUFBRSxDQUFDLFNBQVMsQ0FBQyxvREFBb0QsQ0FBQyxDQUFDO1FBQzFFLE1BQU0sNkJBQTZCLEdBQUcsa0JBQWtCLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFDMUQsa0JBQWtCLENBQUMsQ0FBQztZQUNwQixPQUFPLENBQUMsNkJBQTZCLENBQUM7UUFDMUMsTUFBTSxpQkFBaUIsR0FDbkIsUUFBUSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxHQUFHLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsV0FBVyxHQUFHLEVBQUUsQ0FBQyxDQUFDO1FBQ3pFLE1BQU0sZ0JBQWdCLEdBQ2xCLGlCQUFpQixJQUFJLDZCQUE2QjtZQUNsRCxDQUFDLFdBQVcsSUFBSSxDQUFDO2dCQUNoQixpQkFBaUIsSUFBSSw2QkFBNkIsR0FBRyxDQUFDLENBQUMsQ0FBQztRQUM3RCxJQUFJLGdCQUFnQixFQUFFO1lBQ3BCLElBQUksUUFBUSxHQUFHLFdBQVcsR0FBRyxXQUFXLElBQUksR0FBRyxFQUFFO2dCQUMvQyxpQkFBaUIsR0FBRyxpQkFBaUIsQ0FBQyxtQkFBbUIsQ0FBQzthQUMzRDtpQkFBTSxJQUFJLFFBQVEsS0FBSyxDQUFDLElBQUksV0FBVyxJQUFJLElBQUksRUFBRTtnQkFDaEQsaUJBQWlCLEdBQUcsaUJBQWlCLENBQUMsbUJBQW1CLENBQUM7YUFDM0Q7aUJBQU07Z0JBQ0wsaUJBQWlCLEdBQUcsaUJBQWlCLENBQUMsNEJBQTRCLENBQUM7YUFDcEU7U0FDRjthQUFNO1lBQ0wsaUJBQWlCLEdBQUcsaUJBQWlCLENBQUMsbUJBQW1CLENBQUM7U0FDM0Q7S0FDRjtJQUVELFFBQVEsaUJBQWlCLEVBQUU7UUFDekIsS0FBSyxpQkFBaUIsQ0FBQyxtQkFBbUI7WUFDeEMsT0FBTyxHQUFHLElBQUksbUJBQW1CLENBQzdCLFdBQVcsRUFBRSxVQUFVLEVBQUUsVUFBVSxFQUFFLElBQUksRUFBRSxVQUFVLEVBQ3JELHNCQUFzQixDQUFDLENBQUM7WUFDNUIsTUFBTTtRQUNSLEtBQUssaUJBQWlCLENBQUMsbUJBQW1CLENBQUMsQ0FBQztZQUMxQyxxRUFBcUU7WUFDckUsd0NBQXdDO1lBQ3hDLEdBQUcsR0FBRyxJQUFJLENBQ04sRUFBQyxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLFdBQVcsRUFBRSxLQUFLLEVBQUUsQ0FBQyxFQUFFLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFDLEVBQUMsQ0FBQyxDQUFDO1lBQ3RFLE9BQU8sR0FBRyxJQUFJLG1CQUFtQixDQUM3QixXQUFXLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxVQUFVLENBQUMsQ0FBQztZQUN0RCxJQUFJLElBQUksSUFBSSxVQUFVLEVBQUU7Z0JBQ3RCLEdBQUc7b0JBQ0MsT0FBTyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxNQUFNLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxVQUFVLEVBQUUsR0FBRyxDQUFDLENBQUM7Z0JBQ3hFLE1BQU0scUJBQXFCLEdBQUcsSUFBSSxxQkFBcUIsQ0FDbkQsR0FBRyxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsVUFBVSxFQUFFLHNCQUFzQixDQUFDLENBQUM7Z0JBQ3pELElBQUksV0FBVyxHQUFHLElBQUksQ0FBQztnQkFDdkIsTUFBTSxnQkFBZ0IsR0FBaUIsQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFDN0MsSUFBSSxJQUFJLEVBQUU7b0JBQ1IsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO2lCQUM3QjtnQkFDRCxJQUFJLHNCQUFzQixFQUFFO29CQUMxQixnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsc0JBQXNCLENBQUMsQ0FBQztpQkFDL0M7Z0JBQ0QsSUFBSSxVQUFVLEtBQUssV0FBVyxFQUFFO29CQUM5QixXQUFXLEdBQUcsQ0FBQyxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsY0FBYyxDQUFDLEVBQUMsQ0FBQyxDQUFDO29CQUMxRCxxQkFBcUIsQ0FBQyxRQUFRLElBQUksZUFBZSxDQUFDO2lCQUNuRDtnQkFDRCxNQUFNLFlBQVksR0FBRyxPQUFPLENBQUMsZ0JBQWdCLENBQ3pDLHFCQUFxQixFQUFFLGdCQUFnQixFQUFFLEdBQUcsQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7Z0JBQ3JFLGFBQWEsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBQ3hCLE1BQU0sV0FBVyxHQUFHLE9BQU8sQ0FDdkIsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsWUFBWSxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxRQUFRLEVBQUMsRUFBQyxDQUFDLENBQUM7Z0JBQ3BFLGFBQWEsQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUM7Z0JBQ2pDLEtBQUssTUFBTSxDQUFDLElBQUksYUFBYSxFQUFFO29CQUM3QixPQUFPLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQztpQkFDL0I7Z0JBQ0QsT0FBTyxXQUFXLENBQUM7YUFDcEI7WUFDRCxNQUFNO1NBQ1A7UUFDRCxLQUFLLGlCQUFpQixDQUFDLDRCQUE0QjtZQUNqRCxPQUFPLEdBQUcsSUFBSSw0QkFBNEIsQ0FDdEMsUUFBUSxFQUFFLFFBQVEsRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLFVBQVUsRUFBRSxJQUFJLEVBQzdELFVBQVUsRUFBRSxzQkFBc0IsQ0FBQyxDQUFDO1lBQ3hDLE1BQU07UUFDUixLQUFLLGlCQUFpQixDQUFDLG1CQUFtQjtZQUN4QyxxRUFBcUU7WUFDckUsUUFBUTtZQUNSLE1BQU0seUJBQXlCLEdBQUcsT0FBTyxDQUFDLFdBQVcsQ0FBQyxPQUFPLEVBQUUsQ0FBQztZQUNoRSxPQUFPLEdBQUcsSUFBSSxtQkFBbUIsQ0FDN0IsUUFBUSxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsVUFBVSxFQUFFLElBQUksRUFBRSxVQUFVLEVBQy9ELHNCQUFzQixFQUFFLHlCQUF5QixDQUFDLENBQUM7WUFDdkQsTUFBTTtRQUNSO1lBQ0UsTUFBTSxJQUFJLEtBQUssQ0FBQyxpQ0FBaUMsaUJBQWlCLEdBQUcsQ0FBQyxDQUFDO0tBQzFFO0lBRUQsSUFBSSxJQUFJLEVBQUU7UUFDUixNQUFNLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0tBQ25CO0lBQ0QsSUFBSSxzQkFBc0IsRUFBRTtRQUMxQixNQUFNLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUM7S0FDckM7SUFDRCxJQUFJLFVBQVUsS0FBSyxXQUFXLEVBQUU7UUFDOUIsVUFBVSxDQUFDLElBQUksQ0FBQyxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsY0FBYyxDQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQzNELE9BQU8sQ0FBQyxRQUFRLElBQUksZUFBZSxDQUFDO0tBQ3JDO0lBQ0QsR0FBRyxHQUFHLE9BQU8sQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUUsTUFBTSxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsVUFBVSxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBQzFFLE1BQU0sV0FBVyxHQUNiLE9BQU8sQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxHQUFHLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFFLFFBQVEsRUFBQyxFQUFDLENBQUMsQ0FBQztJQUNuRSxhQUFhLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBQ3hCLEtBQUssTUFBTSxDQUFDLElBQUksYUFBYSxFQUFFO1FBQzdCLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0tBQy9CO0lBQ0QsT0FBTyxXQUFXLENBQUM7QUFDckIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIGJyb2FkY2FzdF91dGlsLCBlbnYsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtNYXRNdWxQYWNrZWRQcm9ncmFtfSBmcm9tICcuLi9tYXRtdWxfcGFja2VkX3dlYmdwdSc7XG5pbXBvcnQge01hdE11bFJlZHVjZVByb2dyYW19IGZyb20gJy4uL21hdG11bF9yZWR1Y2Vfd2ViZ3B1JztcbmltcG9ydCB7TWF0TXVsU21hbGxPdXRwdXRTaXplUHJvZ3JhbX0gZnJvbSAnLi4vbWF0bXVsX3NtYWxsX291dHB1dF9zaXplX3dlYmdwdSc7XG5pbXBvcnQge0JpYXNBY3RpdmF0aW9uUHJvZ3JhbSwgTWF0TXVsU3BsaXRLUHJvZ3JhbX0gZnJvbSAnLi4vbWF0bXVsX3NwbGl0S193ZWJncHUnO1xuaW1wb3J0IHtXZWJHUFVQcm9ncmFtfSBmcm9tICcuLi93ZWJncHVfcHJvZ3JhbSc7XG5pbXBvcnQge01hdE11bFByb2dyYW1UeXBlfSBmcm9tICcuLi93ZWJncHVfdXRpbCc7XG5cbmltcG9ydCB7ZmlsbH0gZnJvbSAnLi9GaWxsJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9SZXNoYXBlJztcblxudHlwZSBCYXRjaE1hdE11bENvbmZpZyA9IHtcbiAgYTogVGVuc29ySW5mbyxcbiAgYjogVGVuc29ySW5mbyxcbiAgdHJhbnNwb3NlQTogYm9vbGVhbixcbiAgdHJhbnNwb3NlQjogYm9vbGVhbixcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZCxcbiAgYmlhcz86IFRlbnNvckluZm8sXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHM/OiBUZW5zb3JJbmZvLFxuICBsZWFreXJlbHVBbHBoYT86IG51bWJlcixcbiAgYWN0aXZhdGlvbj86IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uXG59O1xuXG5leHBvcnQgZnVuY3Rpb24gYmF0Y2hNYXRNdWxJbXBsKHtcbiAgYSxcbiAgYixcbiAgdHJhbnNwb3NlQSxcbiAgdHJhbnNwb3NlQixcbiAgYmFja2VuZCxcbiAgYmlhcyA9IG51bGwsXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBudWxsLFxuICBsZWFreXJlbHVBbHBoYSA9IDAsXG4gIGFjdGl2YXRpb24gPSBudWxsXG59OiBCYXRjaE1hdE11bENvbmZpZyk6IFRlbnNvckluZm8ge1xuICBjb25zdCBhUmFuayA9IGEuc2hhcGUubGVuZ3RoO1xuICBjb25zdCBiUmFuayA9IGIuc2hhcGUubGVuZ3RoO1xuXG4gIGNvbnN0IGlubmVyU2hhcGVBID0gdHJhbnNwb3NlQSA/IGEuc2hhcGVbYVJhbmsgLSAyXSA6IGEuc2hhcGVbYVJhbmsgLSAxXTtcbiAgY29uc3QgaW5uZXJTaGFwZUIgPSB0cmFuc3Bvc2VCID8gYi5zaGFwZVtiUmFuayAtIDFdIDogYi5zaGFwZVtiUmFuayAtIDJdO1xuXG4gIGNvbnN0IG91dGVyU2hhcGVBID0gdHJhbnNwb3NlQSA/IGEuc2hhcGVbYVJhbmsgLSAxXSA6IGEuc2hhcGVbYVJhbmsgLSAyXTtcbiAgY29uc3Qgb3V0ZXJTaGFwZUIgPSB0cmFuc3Bvc2VCID8gYi5zaGFwZVtiUmFuayAtIDJdIDogYi5zaGFwZVtiUmFuayAtIDFdO1xuXG4gIGNvbnN0IG91dGVyRGltc0EgPSBhLnNoYXBlLnNsaWNlKDAsIC0yKTtcbiAgY29uc3Qgb3V0ZXJEaW1zQiA9IGIuc2hhcGUuc2xpY2UoMCwgLTIpO1xuXG4gIGNvbnN0IGJhdGNoRGltQSA9IHV0aWwuc2l6ZUZyb21TaGFwZShvdXRlckRpbXNBKTtcbiAgY29uc3QgYmF0Y2hEaW1CID0gdXRpbC5zaXplRnJvbVNoYXBlKG91dGVyRGltc0IpO1xuXG4gIGNvbnN0IG91dFNoYXBlT3V0ZXJEaW1zID0gYnJvYWRjYXN0X3V0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGUoXG4gICAgICBhLnNoYXBlLnNsaWNlKDAsIC0yKSwgYi5zaGFwZS5zbGljZSgwLCAtMikpO1xuICBjb25zdCBvdXRTaGFwZSA9IG91dFNoYXBlT3V0ZXJEaW1zLmNvbmNhdChbb3V0ZXJTaGFwZUEsIG91dGVyU2hhcGVCXSk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICBpbm5lclNoYXBlQSA9PT0gaW5uZXJTaGFwZUIsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gbWF0TXVsOiBpbm5lciBzaGFwZXMgKCR7aW5uZXJTaGFwZUF9KSBhbmQgKGAgK1xuICAgICAgICAgIGAke2lubmVyU2hhcGVCfSkgb2YgVGVuc29ycyB3aXRoIHNoYXBlcyAke2Euc2hhcGV9IGFuZCBgICtcbiAgICAgICAgICBgJHtiLnNoYXBlfSBhbmQgdHJhbnNwb3NlQT0ke3RyYW5zcG9zZUF9YCArXG4gICAgICAgICAgYCBhbmQgdHJhbnNwb3NlQj0ke3RyYW5zcG9zZUJ9IG11c3QgbWF0Y2guYCk7XG5cbiAgY29uc3QgYTNkU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IHRyYW5zcG9zZUEgP1xuICAgICAgW2JhdGNoRGltQSwgaW5uZXJTaGFwZUEsIG91dGVyU2hhcGVBXSA6XG4gICAgICBbYmF0Y2hEaW1BLCBvdXRlclNoYXBlQSwgaW5uZXJTaGFwZUFdO1xuICBjb25zdCBiM2RTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gdHJhbnNwb3NlQiA/XG4gICAgICBbYmF0Y2hEaW1CLCBvdXRlclNoYXBlQiwgaW5uZXJTaGFwZUJdIDpcbiAgICAgIFtiYXRjaERpbUIsIGlubmVyU2hhcGVCLCBvdXRlclNoYXBlQl07XG5cbiAgLy8gVGhlIHJlc3Qgb2YgdGhlIGltcGxlbWVudGF0aW9uIGlzIGRlc2lnbmVkIHRvIG9wZXJhdGUgb24gcmFuay0zIHRlbnNvcnNcbiAgY29uc3QgYTNkID0gcmVzaGFwZSh7aW5wdXRzOiB7eDogYX0sIGJhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IGEzZFNoYXBlfX0pO1xuICBjb25zdCBiM2QgPSByZXNoYXBlKHtpbnB1dHM6IHt4OiBifSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogYjNkU2hhcGV9fSk7XG4gIGNvbnN0IGludGVybWVkaWF0ZXM6IFRlbnNvckluZm9bXSA9IFthM2QsIGIzZF07XG5cbiAgY29uc3QgYmF0Y2hEaW0gPSBNYXRoLm1heChiYXRjaERpbUEsIGJhdGNoRGltQik7XG5cbiAgY29uc3QgaW5wdXRzOiBUZW5zb3JJbmZvW10gPSBbYTNkLCBiM2RdO1xuICBjb25zdCBkaW1lbnNpb25zID0gW1xuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbb3V0ZXJTaGFwZUFdfSwge3R5cGU6ICdpbnQzMicsIGRhdGE6IFtvdXRlclNoYXBlQl19LFxuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbaW5uZXJTaGFwZUFdfVxuICBdO1xuXG4gIGxldCBwcm9ncmFtOiBXZWJHUFVQcm9ncmFtO1xuICBsZXQgb3V0OiBUZW5zb3JJbmZvO1xuICBjb25zdCBvdXRwdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID1cbiAgICAgIFtiYXRjaERpbSwgb3V0ZXJTaGFwZUEsIG91dGVyU2hhcGVCXTtcbiAgbGV0IG1hdG11bFByb2dyYW1UeXBlID0gZW52KCkuZ2V0KCdXRUJHUFVfTUFUTVVMX1BST0dSQU1fVFlQRScpIGFzIG51bWJlcjtcbiAgaWYgKG1hdG11bFByb2dyYW1UeXBlIDwgMCkge1xuICAgIC8vIFVzdWFsbHkgaW5jcmVhc2luZyB3b3JrZ3JvdXBzIGlzIGEgZ29vZCB3YXkgdG8gZ2FpbiBtb3JlIHBlcmZvcm1hbmNlIGZvclxuICAgIC8vIGZldyB3b3JrZ3JvdXBzIGJ5IHRpbGluZyAzMngzMiAoZGVmYXVsdCBtYXRtdWwgYWxnb3JpdGhtKS4gQ3VycmVudGx5LFxuICAgIC8vIHRoZXJlIGFyZSB0aHJlZSB3YXlzIHRvIGluY3JlYXNlIHdvcmtncm91cHMuIDEpIE1hdE11bFJlZHVjZVByb2dyYW0sXG4gICAgLy8gd2hpY2ggaXMgdXNlZCBvbmx5IHdoZW4gdGhlIG91dHB1dCBzaXplIGlzIHZlcnkgc21hbGwgKDEyOCBmb3Igbm93KS4gMilcbiAgICAvLyBNYXRNdWxTcGxpdEtQcm9ncmFtLCBpbmNyZWFzaW5nIHdvcmtncm91cHMgYnkgc3BsaXRpbmcgSy4gMylcbiAgICAvLyBNYXRNdWxTbWFsbE91dHB1dFNpemVQcm9ncmFtLCBpbmNyZWFzaW5nIHdvcmtncm91cHMgYnkgc21hbGwgdGlsZSBzaXplLlxuICAgIC8vIEZvciBkaWZmZXJlbnQgZGV2aWNlcywgdGhlIG1pbmltdW0gb3B0aW1hbCB3b3JrZ3JvdXBzIG1heSBiZSBkaWZmZXJlbnQuXG4gICAgLy8gU28gaGVyZSB3ZSBzZXQgYSB8dGhyZXNob2xkVG9JbmNyZWFzZVdvcmtncm91cHN8IHRvIGluZGljYXRlIHdoZXRoZXIgd2VcbiAgICAvLyBuZWVkIHRvIGluY3JlYXNlIHdvcmtncm91cHMuIEFuZCB0aGUgbGl0ZXJhbCBudW1iZXIgaXMgYW4gZW1waXJpY2FsXG4gICAgLy8gdmFsdWUuXG4gICAgY29uc3QgdGhyZXNob2xkRmxhZ1ZhbHVlID1cbiAgICAgICAgZW52KCkuZ2V0TnVtYmVyKCdXRUJHUFVfVEhSRVNIT0xEX1RPX0lOQ1JFQVNFX1dPUktHUk9VUFNfRk9SX01BVE1VTCcpO1xuICAgIGNvbnN0IHRocmVzaG9sZFRvSW5jcmVhc2VXb3JrZ3JvdXBzID0gdGhyZXNob2xkRmxhZ1ZhbHVlID4gMCA/XG4gICAgICAgIHRocmVzaG9sZEZsYWdWYWx1ZSA6XG4gICAgICAgIGJhY2tlbmQudGhyZXNob2xkVG9JbmNyZWFzZVdvcmtncm91cHM7XG4gICAgY29uc3Qgd29ya2dyb3Vwc0J5MzJ4MzIgPVxuICAgICAgICBiYXRjaERpbSAqIE1hdGguY2VpbChvdXRlclNoYXBlQSAvIDMyKSAqIE1hdGguY2VpbChvdXRlclNoYXBlQiAvIDMyKTtcbiAgICBjb25zdCBoYXNGZXdXb3JrZ3JvdXBzID1cbiAgICAgICAgd29ya2dyb3Vwc0J5MzJ4MzIgPD0gdGhyZXNob2xkVG9JbmNyZWFzZVdvcmtncm91cHMgfHxcbiAgICAgICAgKG91dGVyU2hhcGVBIDw9IDggJiZcbiAgICAgICAgIHdvcmtncm91cHNCeTMyeDMyIDw9IHRocmVzaG9sZFRvSW5jcmVhc2VXb3JrZ3JvdXBzICogMik7XG4gICAgaWYgKGhhc0Zld1dvcmtncm91cHMpIHtcbiAgICAgIGlmIChiYXRjaERpbSAqIG91dGVyU2hhcGVBICogb3V0ZXJTaGFwZUIgPD0gMTI4KSB7XG4gICAgICAgIG1hdG11bFByb2dyYW1UeXBlID0gTWF0TXVsUHJvZ3JhbVR5cGUuTWF0TXVsUmVkdWNlUHJvZ3JhbTtcbiAgICAgIH0gZWxzZSBpZiAoYmF0Y2hEaW0gPT09IDEgJiYgaW5uZXJTaGFwZUIgPj0gMjAwMCkge1xuICAgICAgICBtYXRtdWxQcm9ncmFtVHlwZSA9IE1hdE11bFByb2dyYW1UeXBlLk1hdE11bFNwbGl0S1Byb2dyYW07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBtYXRtdWxQcm9ncmFtVHlwZSA9IE1hdE11bFByb2dyYW1UeXBlLk1hdE11bFNtYWxsT3V0cHV0U2l6ZVByb2dyYW07XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIG1hdG11bFByb2dyYW1UeXBlID0gTWF0TXVsUHJvZ3JhbVR5cGUuTWF0TXVsUGFja2VkUHJvZ3JhbTtcbiAgICB9XG4gIH1cblxuICBzd2l0Y2ggKG1hdG11bFByb2dyYW1UeXBlKSB7XG4gICAgY2FzZSBNYXRNdWxQcm9ncmFtVHlwZS5NYXRNdWxSZWR1Y2VQcm9ncmFtOlxuICAgICAgcHJvZ3JhbSA9IG5ldyBNYXRNdWxSZWR1Y2VQcm9ncmFtKFxuICAgICAgICAgIG91dHB1dFNoYXBlLCB0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCLCBiaWFzLCBhY3RpdmF0aW9uLFxuICAgICAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSBNYXRNdWxQcm9ncmFtVHlwZS5NYXRNdWxTcGxpdEtQcm9ncmFtOiB7XG4gICAgICAvLyBUaGUgb3V0cHV0IGJ1ZmZlciBtdXN0IGJlIGluaXRhaWx6ZWQgdG8gemVybyBiZWZvcmUgdXNpbmcgc2luY2Ugd2VcbiAgICAgIC8vIHVzZSBhdG9taWNBZGQgaW4gTWF0TXVsU3BsaXRLUHJvZ3JhbS5cbiAgICAgIG91dCA9IGZpbGwoXG4gICAgICAgICAge2JhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IG91dHB1dFNoYXBlLCB2YWx1ZTogMCwgZHR5cGU6IGEuZHR5cGV9fSk7XG4gICAgICBwcm9ncmFtID0gbmV3IE1hdE11bFNwbGl0S1Byb2dyYW0oXG4gICAgICAgICAgb3V0cHV0U2hhcGUsIGlubmVyU2hhcGVCLCB0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCKTtcbiAgICAgIGlmIChiaWFzIHx8IGFjdGl2YXRpb24pIHtcbiAgICAgICAgb3V0ID1cbiAgICAgICAgICAgIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBpbnB1dHMsIGEuZHR5cGUsIGRpbWVuc2lvbnMsIG91dCk7XG4gICAgICAgIGNvbnN0IGJpYXNBY3RpdmF0aW9uUHJvZ3JhbSA9IG5ldyBCaWFzQWN0aXZhdGlvblByb2dyYW0oXG4gICAgICAgICAgICBvdXQuc2hhcGUsIGJpYXMsIGFjdGl2YXRpb24sIHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICAgICAgICBsZXQgdW5pZm9ybURhdGEgPSBudWxsO1xuICAgICAgICBjb25zdCBhY3RpdmF0aW9uSW5wdXRzOiBUZW5zb3JJbmZvW10gPSBbb3V0XTtcbiAgICAgICAgaWYgKGJpYXMpIHtcbiAgICAgICAgICBhY3RpdmF0aW9uSW5wdXRzLnB1c2goYmlhcyk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKHByZWx1QWN0aXZhdGlvbldlaWdodHMpIHtcbiAgICAgICAgICBhY3RpdmF0aW9uSW5wdXRzLnB1c2gocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGFjdGl2YXRpb24gPT09ICdsZWFreXJlbHUnKSB7XG4gICAgICAgICAgdW5pZm9ybURhdGEgPSBbe3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2xlYWt5cmVsdUFscGhhXX1dO1xuICAgICAgICAgIGJpYXNBY3RpdmF0aW9uUHJvZ3JhbS51bmlmb3JtcyArPSAnIGFscGhhIDogZjMyLCc7XG4gICAgICAgIH1cbiAgICAgICAgY29uc3Qgb3V0QWN0aXZhdGVkID0gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKFxuICAgICAgICAgICAgYmlhc0FjdGl2YXRpb25Qcm9ncmFtLCBhY3RpdmF0aW9uSW5wdXRzLCBvdXQuZHR5cGUsIHVuaWZvcm1EYXRhKTtcbiAgICAgICAgaW50ZXJtZWRpYXRlcy5wdXNoKG91dCk7XG4gICAgICAgIGNvbnN0IG91dFJlc2hhcGVkID0gcmVzaGFwZShcbiAgICAgICAgICAgIHtpbnB1dHM6IHt4OiBvdXRBY3RpdmF0ZWR9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiBvdXRTaGFwZX19KTtcbiAgICAgICAgaW50ZXJtZWRpYXRlcy5wdXNoKG91dEFjdGl2YXRlZCk7XG4gICAgICAgIGZvciAoY29uc3QgaSBvZiBpbnRlcm1lZGlhdGVzKSB7XG4gICAgICAgICAgYmFja2VuZC5kaXNwb3NlRGF0YShpLmRhdGFJZCk7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIG91dFJlc2hhcGVkO1xuICAgICAgfVxuICAgICAgYnJlYWs7XG4gICAgfVxuICAgIGNhc2UgTWF0TXVsUHJvZ3JhbVR5cGUuTWF0TXVsU21hbGxPdXRwdXRTaXplUHJvZ3JhbTpcbiAgICAgIHByb2dyYW0gPSBuZXcgTWF0TXVsU21hbGxPdXRwdXRTaXplUHJvZ3JhbShcbiAgICAgICAgICBhM2RTaGFwZSwgYjNkU2hhcGUsIG91dHB1dFNoYXBlLCB0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCLCBiaWFzLFxuICAgICAgICAgIGFjdGl2YXRpb24sIHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSBNYXRNdWxQcm9ncmFtVHlwZS5NYXRNdWxQYWNrZWRQcm9ncmFtOlxuICAgICAgLy8gRXhwZXJpbWVudHMgc2hvdyB0aGF0IHNlcXVlbnRpYWwgYWNjZXNzIGlzIG1vcmUgZnJpZW5kbHkgZm9yIEludGVsXG4gICAgICAvLyBHUFVzLlxuICAgICAgY29uc3Qgc2VxdWVudGlhbEFjY2Vzc0J5VGhyZWFkcyA9IGJhY2tlbmQuYWRhcHRlckluZm8uaXNJbnRlbCgpO1xuICAgICAgcHJvZ3JhbSA9IG5ldyBNYXRNdWxQYWNrZWRQcm9ncmFtKFxuICAgICAgICAgIGEzZFNoYXBlLCBvdXRwdXRTaGFwZSwgdHJhbnNwb3NlQSwgdHJhbnNwb3NlQiwgYmlhcywgYWN0aXZhdGlvbixcbiAgICAgICAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLCBzZXF1ZW50aWFsQWNjZXNzQnlUaHJlYWRzKTtcbiAgICAgIGJyZWFrO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIE1hdE11bFByb2dyYW1UeXBlICR7bWF0bXVsUHJvZ3JhbVR5cGV9LmApO1xuICB9XG5cbiAgaWYgKGJpYXMpIHtcbiAgICBpbnB1dHMucHVzaChiaWFzKTtcbiAgfVxuICBpZiAocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cykge1xuICAgIGlucHV0cy5wdXNoKHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICB9XG4gIGlmIChhY3RpdmF0aW9uID09PSAnbGVha3lyZWx1Jykge1xuICAgIGRpbWVuc2lvbnMucHVzaCh7dHlwZTogJ2Zsb2F0MzInLCBkYXRhOiBbbGVha3lyZWx1QWxwaGFdfSk7XG4gICAgcHJvZ3JhbS51bmlmb3JtcyArPSAnIGFscGhhIDogZjMyLCc7XG4gIH1cbiAgb3V0ID0gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKHByb2dyYW0sIGlucHV0cywgYS5kdHlwZSwgZGltZW5zaW9ucywgb3V0KTtcbiAgY29uc3Qgb3V0UmVzaGFwZWQgPVxuICAgICAgcmVzaGFwZSh7aW5wdXRzOiB7eDogb3V0fSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogb3V0U2hhcGV9fSk7XG4gIGludGVybWVkaWF0ZXMucHVzaChvdXQpO1xuICBmb3IgKGNvbnN0IGkgb2YgaW50ZXJtZWRpYXRlcykge1xuICAgIGJhY2tlbmQuZGlzcG9zZURhdGEoaS5kYXRhSWQpO1xuICB9XG4gIHJldHVybiBvdXRSZXNoYXBlZDtcbn1cbiJdfQ==