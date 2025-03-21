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
import { Complex } from '@tensorflow/tfjs-core';
import { identity } from './Identity';
/**
 * Complex tensors share data with their real and imaginary components. Complex
 * tensors' reference to the components is tracked by refCount on the individual
 * component. The refCounts are increased by the identity call.
 *
 * When a complex tensor is disposed, it will reduce the refCount on the
 * components by calling disposeData on each.
 */
export function complex(args) {
    const { inputs, backend } = args;
    const { real, imag } = inputs;
    const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
    const complex = backend.tensorMap.get(complexInfo.dataId);
    const realTensorInfo = identity({ inputs: { x: real }, backend });
    const imagTensorInfo = identity({ inputs: { x: imag }, backend });
    complex.complexTensorInfos = { real: realTensorInfo, imag: imagTensorInfo };
    return complexInfo;
}
export const complexConfig = {
    kernelName: Complex,
    backendName: 'webgpu',
    kernelFunc: complex
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29tcGxleC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvQ29tcGxleC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsT0FBTyxFQUFzRCxNQUFNLHVCQUF1QixDQUFDO0FBR25HLE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFcEM7Ozs7Ozs7R0FPRztBQUNILE1BQU0sVUFBVSxPQUFPLENBQUMsSUFBcUQ7SUFFM0UsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDL0IsTUFBTSxFQUFDLElBQUksRUFBRSxJQUFJLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFFNUIsTUFBTSxXQUFXLEdBQUcsT0FBTyxDQUFDLGNBQWMsQ0FBQyxJQUFJLENBQUMsS0FBSyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQ3BFLE1BQU0sT0FBTyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUUxRCxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsSUFBSSxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztJQUU5RCxNQUFNLGNBQWMsR0FBRyxRQUFRLENBQUMsRUFBQyxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUUsSUFBSSxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztJQUU5RCxPQUFPLENBQUMsa0JBQWtCLEdBQUcsRUFBQyxJQUFJLEVBQUUsY0FBYyxFQUFFLElBQUksRUFBRSxjQUFjLEVBQUMsQ0FBQztJQUUxRSxPQUFPLFdBQVcsQ0FBQztBQUNyQixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFpQjtJQUN6QyxVQUFVLEVBQUUsT0FBTztJQUNuQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsT0FBZ0M7Q0FDN0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtDb21wbGV4LCBDb21wbGV4SW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtpZGVudGl0eX0gZnJvbSAnLi9JZGVudGl0eSc7XG5cbi8qKlxuICogQ29tcGxleCB0ZW5zb3JzIHNoYXJlIGRhdGEgd2l0aCB0aGVpciByZWFsIGFuZCBpbWFnaW5hcnkgY29tcG9uZW50cy4gQ29tcGxleFxuICogdGVuc29ycycgcmVmZXJlbmNlIHRvIHRoZSBjb21wb25lbnRzIGlzIHRyYWNrZWQgYnkgcmVmQ291bnQgb24gdGhlIGluZGl2aWR1YWxcbiAqIGNvbXBvbmVudC4gVGhlIHJlZkNvdW50cyBhcmUgaW5jcmVhc2VkIGJ5IHRoZSBpZGVudGl0eSBjYWxsLlxuICpcbiAqIFdoZW4gYSBjb21wbGV4IHRlbnNvciBpcyBkaXNwb3NlZCwgaXQgd2lsbCByZWR1Y2UgdGhlIHJlZkNvdW50IG9uIHRoZVxuICogY29tcG9uZW50cyBieSBjYWxsaW5nIGRpc3Bvc2VEYXRhIG9uIGVhY2guXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjb21wbGV4KGFyZ3M6IHtpbnB1dHM6IENvbXBsZXhJbnB1dHMsIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmR9KTpcbiAgICBUZW5zb3JJbmZvIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZH0gPSBhcmdzO1xuICBjb25zdCB7cmVhbCwgaW1hZ30gPSBpbnB1dHM7XG5cbiAgY29uc3QgY29tcGxleEluZm8gPSBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKHJlYWwuc2hhcGUsICdjb21wbGV4NjQnKTtcbiAgY29uc3QgY29tcGxleCA9IGJhY2tlbmQudGVuc29yTWFwLmdldChjb21wbGV4SW5mby5kYXRhSWQpO1xuXG4gIGNvbnN0IHJlYWxUZW5zb3JJbmZvID0gaWRlbnRpdHkoe2lucHV0czoge3g6IHJlYWx9LCBiYWNrZW5kfSk7XG5cbiAgY29uc3QgaW1hZ1RlbnNvckluZm8gPSBpZGVudGl0eSh7aW5wdXRzOiB7eDogaW1hZ30sIGJhY2tlbmR9KTtcblxuICBjb21wbGV4LmNvbXBsZXhUZW5zb3JJbmZvcyA9IHtyZWFsOiByZWFsVGVuc29ySW5mbywgaW1hZzogaW1hZ1RlbnNvckluZm99O1xuXG4gIHJldHVybiBjb21wbGV4SW5mbztcbn1cblxuZXhwb3J0IGNvbnN0IGNvbXBsZXhDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQ29tcGxleCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBjb21wbGV4IGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==