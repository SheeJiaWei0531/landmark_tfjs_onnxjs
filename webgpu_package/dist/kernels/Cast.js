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
import * as tf from '@tensorflow/tfjs-core';
import { Cast, util } from '@tensorflow/tfjs-core';
import { castImplCPU } from '../kernel_utils/shared';
import { complex } from './Complex';
import { identity } from './Identity';
import { notEqual } from './NotEqual';
import { real } from './Real';
import { int } from '../kernel_utils/int';
export function cast(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dtype } = attrs;
    // Casting to complex64.
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return identity({ inputs: { x }, backend });
        }
        // TODO: Import kernel function once zeros is modularized.
        const zerosTensor = tf.zeros(x.shape);
        const floatX = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
        const result = complex({ inputs: { real: floatX, imag: zerosTensor }, backend });
        zerosTensor.dispose();
        backend.disposeData(floatX.dataId);
        return result;
    }
    // Casting from complex64
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const result = cast({ inputs: { x: realPart }, backend, attrs: { dtype } });
        backend.disposeData(realPart.dataId);
        return result;
    }
    if (!util.hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        const result = identity({ inputs: { x }, backend });
        return { dataId: result.dataId, shape: result.shape, dtype };
    }
    if (backend.shouldExecuteOnCPU([x])) {
        const values = backend.tensorMap.get(x.dataId).values;
        const [resultShape, resultType, resultData] = castImplCPU(values, x.shape, x.dtype, dtype);
        return backend.makeTensorInfo(resultShape, resultType, resultData);
    }
    if (dtype === 'int32') {
        return int(x, backend);
    }
    if (dtype === 'bool') {
        const zerosTensorInfo = backend.makeTensorInfo([], 'bool', util.getTypedArrayFromDType('bool', 1));
        const binaryInputs = { a: x, b: zerosTensorInfo };
        const result = notEqual({ inputs: binaryInputs, backend });
        backend.disposeData(zerosTensorInfo.dataId);
        return result;
    }
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}
export const castConfig = {
    kernelName: Cast,
    backendName: 'webgpu',
    kernelFunc: cast
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ2FzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvQ2FzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEtBQUssRUFBRSxNQUFNLHVCQUF1QixDQUFDO0FBQzVDLE9BQU8sRUFBZSxJQUFJLEVBQTJFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBR3hJLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSx3QkFBd0IsQ0FBQztBQUVuRCxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2xDLE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDcEMsT0FBTyxFQUFDLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNwQyxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBRTVCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUV4QyxNQUFNLFVBQVUsSUFBSSxDQUNoQixJQUFvRTtJQUV0RSxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUNuQixNQUFNLEVBQUMsS0FBSyxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRXRCLHdCQUF3QjtJQUN4QixJQUFJLEtBQUssS0FBSyxXQUFXLEVBQUU7UUFDekIsSUFBSSxDQUFDLENBQUMsS0FBSyxLQUFLLFdBQVcsRUFBRTtZQUMzQixPQUFPLFFBQVEsQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBQyxFQUFFLE9BQU8sRUFBQyxDQUFDLENBQUM7U0FDekM7UUFFRCwwREFBMEQ7UUFDMUQsTUFBTSxXQUFXLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDdEMsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxTQUFTLEVBQUMsRUFBQyxDQUFDLENBQUM7UUFFdkUsTUFBTSxNQUFNLEdBQ1IsT0FBTyxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsV0FBVyxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztRQUVsRSxXQUFXLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDdEIsT0FBTyxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFbkMsT0FBTyxNQUFNLENBQUM7S0FDZjtJQUVELHlCQUF5QjtJQUN6QixJQUFJLENBQUMsQ0FBQyxLQUFLLEtBQUssV0FBVyxFQUFFO1FBQzNCLE1BQU0sUUFBUSxHQUFHLElBQUksQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLEtBQUssRUFBRSxDQUFDLEVBQUMsRUFBRSxPQUFPLEVBQUMsQ0FBQyxDQUFDO1FBQ3JELE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxFQUFDLE1BQU0sRUFBRSxFQUFDLENBQUMsRUFBRSxRQUFRLEVBQUMsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLEVBQUMsS0FBSyxFQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQ3RFLE9BQU8sQ0FBQyxXQUFXLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3JDLE9BQU8sTUFBTSxDQUFDO0tBQ2Y7SUFFRCxJQUFJLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxFQUFFO1FBQ3pDLCtEQUErRDtRQUMvRCxhQUFhO1FBQ2IsTUFBTSxNQUFNLEdBQUcsUUFBUSxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztRQUNoRCxPQUFPLEVBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFDLENBQUM7S0FDNUQ7SUFFRCxJQUFJLE9BQU8sQ0FBQyxrQkFBa0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUU7UUFDbkMsTUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLE1BQW9CLENBQUM7UUFDcEUsTUFBTSxDQUFDLFdBQVcsRUFBRSxVQUFVLEVBQUUsVUFBVSxDQUFDLEdBQ3ZDLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBQ2pELE9BQU8sT0FBTyxDQUFDLGNBQWMsQ0FBQyxXQUFXLEVBQUUsVUFBVSxFQUFFLFVBQVUsQ0FBQyxDQUFDO0tBQ3BFO0lBRUQsSUFBSSxLQUFLLEtBQUssT0FBTyxFQUFFO1FBQ3JCLE9BQU8sR0FBRyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztLQUN4QjtJQUVELElBQUksS0FBSyxLQUFLLE1BQU0sRUFBRTtRQUNwQixNQUFNLGVBQWUsR0FBRyxPQUFPLENBQUMsY0FBYyxDQUMxQyxFQUFFLEVBQUUsTUFBTSxFQUFFLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV4RCxNQUFNLFlBQVksR0FBaUIsRUFBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxlQUFlLEVBQUMsQ0FBQztRQUU5RCxNQUFNLE1BQU0sR0FBRyxRQUFRLENBQUMsRUFBQyxNQUFNLEVBQUUsWUFBWSxFQUFFLE9BQU8sRUFBQyxDQUFlLENBQUM7UUFDdkUsT0FBTyxDQUFDLFdBQVcsQ0FBQyxlQUFlLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDNUMsT0FBTyxNQUFNLENBQUM7S0FDZjtJQUVELE1BQU0sSUFBSSxLQUFLLENBQUMsaUNBQWlDLENBQUMsQ0FBQyxLQUFLLE9BQU8sS0FBSyxFQUFFLENBQUMsQ0FBQztBQUMxRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sVUFBVSxHQUFpQjtJQUN0QyxVQUFVLEVBQUUsSUFBSTtJQUNoQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsSUFBNkI7Q0FDMUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCAqIGFzIHRmIGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge0JpbmFyeUlucHV0cywgQ2FzdCwgQ2FzdEF0dHJzLCBDYXN0SW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtjYXN0SW1wbENQVX0gZnJvbSAnLi4va2VybmVsX3V0aWxzL3NoYXJlZCc7XG5cbmltcG9ydCB7Y29tcGxleH0gZnJvbSAnLi9Db21wbGV4JztcbmltcG9ydCB7aWRlbnRpdHl9IGZyb20gJy4vSWRlbnRpdHknO1xuaW1wb3J0IHtub3RFcXVhbH0gZnJvbSAnLi9Ob3RFcXVhbCc7XG5pbXBvcnQge3JlYWx9IGZyb20gJy4vUmVhbCc7XG5cbmltcG9ydCB7aW50fSBmcm9tICcuLi9rZXJuZWxfdXRpbHMvaW50JztcblxuZXhwb3J0IGZ1bmN0aW9uIGNhc3QoXG4gICAgYXJnczoge2lucHV0czogQ2FzdElucHV0cywgYmFja2VuZDogV2ViR1BVQmFja2VuZCwgYXR0cnM6IENhc3RBdHRyc30pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtkdHlwZX0gPSBhdHRycztcblxuICAvLyBDYXN0aW5nIHRvIGNvbXBsZXg2NC5cbiAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgIGlmICh4LmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgcmV0dXJuIGlkZW50aXR5KHtpbnB1dHM6IHt4fSwgYmFja2VuZH0pO1xuICAgIH1cblxuICAgIC8vIFRPRE86IEltcG9ydCBrZXJuZWwgZnVuY3Rpb24gb25jZSB6ZXJvcyBpcyBtb2R1bGFyaXplZC5cbiAgICBjb25zdCB6ZXJvc1RlbnNvciA9IHRmLnplcm9zKHguc2hhcGUpO1xuICAgIGNvbnN0IGZsb2F0WCA9IGNhc3Qoe2lucHV0czoge3h9LCBiYWNrZW5kLCBhdHRyczoge2R0eXBlOiAnZmxvYXQzMid9fSk7XG5cbiAgICBjb25zdCByZXN1bHQgPVxuICAgICAgICBjb21wbGV4KHtpbnB1dHM6IHtyZWFsOiBmbG9hdFgsIGltYWc6IHplcm9zVGVuc29yfSwgYmFja2VuZH0pO1xuXG4gICAgemVyb3NUZW5zb3IuZGlzcG9zZSgpO1xuICAgIGJhY2tlbmQuZGlzcG9zZURhdGEoZmxvYXRYLmRhdGFJZCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgLy8gQ2FzdGluZyBmcm9tIGNvbXBsZXg2NFxuICBpZiAoeC5kdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICBjb25zdCByZWFsUGFydCA9IHJlYWwoe2lucHV0czoge2lucHV0OiB4fSwgYmFja2VuZH0pO1xuICAgIGNvbnN0IHJlc3VsdCA9IGNhc3Qoe2lucHV0czoge3g6IHJlYWxQYXJ0fSwgYmFja2VuZCwgYXR0cnM6IHtkdHlwZX19KTtcbiAgICBiYWNrZW5kLmRpc3Bvc2VEYXRhKHJlYWxQYXJ0LmRhdGFJZCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIGlmICghdXRpbC5oYXNFbmNvZGluZ0xvc3MoeC5kdHlwZSwgZHR5cGUpKSB7XG4gICAgLy8gV2UgZG9uJ3QgY2hhbmdlIHRoZSB1bmRlcmx5aW5nIGRhdGEsIHNpbmNlIHdlIGNhc3QgdG8gaGlnaGVyXG4gICAgLy8gcHJlY2lzaW9uLlxuICAgIGNvbnN0IHJlc3VsdCA9IGlkZW50aXR5KHtpbnB1dHM6IHt4fSwgYmFja2VuZH0pO1xuICAgIHJldHVybiB7ZGF0YUlkOiByZXN1bHQuZGF0YUlkLCBzaGFwZTogcmVzdWx0LnNoYXBlLCBkdHlwZX07XG4gIH1cblxuICBpZiAoYmFja2VuZC5zaG91bGRFeGVjdXRlT25DUFUoW3hdKSkge1xuICAgIGNvbnN0IHZhbHVlcyA9IGJhY2tlbmQudGVuc29yTWFwLmdldCh4LmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG4gICAgY29uc3QgW3Jlc3VsdFNoYXBlLCByZXN1bHRUeXBlLCByZXN1bHREYXRhXSA9XG4gICAgICAgIGNhc3RJbXBsQ1BVKHZhbHVlcywgeC5zaGFwZSwgeC5kdHlwZSwgZHR5cGUpO1xuICAgIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKHJlc3VsdFNoYXBlLCByZXN1bHRUeXBlLCByZXN1bHREYXRhKTtcbiAgfVxuXG4gIGlmIChkdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIHJldHVybiBpbnQoeCwgYmFja2VuZCk7XG4gIH1cblxuICBpZiAoZHR5cGUgPT09ICdib29sJykge1xuICAgIGNvbnN0IHplcm9zVGVuc29ySW5mbyA9IGJhY2tlbmQubWFrZVRlbnNvckluZm8oXG4gICAgICAgIFtdLCAnYm9vbCcsIHV0aWwuZ2V0VHlwZWRBcnJheUZyb21EVHlwZSgnYm9vbCcsIDEpKTtcblxuICAgIGNvbnN0IGJpbmFyeUlucHV0czogQmluYXJ5SW5wdXRzID0ge2E6IHgsIGI6IHplcm9zVGVuc29ySW5mb307XG5cbiAgICBjb25zdCByZXN1bHQgPSBub3RFcXVhbCh7aW5wdXRzOiBiaW5hcnlJbnB1dHMsIGJhY2tlbmR9KSBhcyBUZW5zb3JJbmZvO1xuICAgIGJhY2tlbmQuZGlzcG9zZURhdGEoemVyb3NUZW5zb3JJbmZvLmRhdGFJZCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHRocm93IG5ldyBFcnJvcihgRXJyb3IgaW4gQ2FzdDogZmFpbGVkIHRvIGNhc3QgJHt4LmR0eXBlfSB0byAke2R0eXBlfWApO1xufVxuXG5leHBvcnQgY29uc3QgY2FzdENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBDYXN0LFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGNhc3QgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19