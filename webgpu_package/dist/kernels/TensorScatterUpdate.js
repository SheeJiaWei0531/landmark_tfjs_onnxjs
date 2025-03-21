/**
 * @license
 * Copyright 2023 Google LLC.
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
import { backend_util, TensorScatterUpdate, util } from '@tensorflow/tfjs-core';
import { ScatterProgram } from '../scatter_webgpu';
import { reshape } from './Reshape';
import { tile } from './Tile';
export function tensorScatterUpdate(args) {
    const { inputs, backend, attrs } = args;
    const { tensor, indices, updates } = inputs;
    const {} = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = backend_util.calculateShapes(updates, indices, tensor.shape);
    const flattenShape = [outputSize / sliceSize, sliceSize];
    if (outputSize === 0) {
        return backend.makeTensorInfo(tensor.shape, indices.dtype);
    }
    const toDispose = [];
    const flattenIndices = reshape({ inputs: { x: indices }, backend, attrs: { shape: [numUpdates, sliceRank] } });
    toDispose.push(flattenIndices);
    const flattenX = reshape({ inputs: { x: updates }, backend, attrs: { shape: [numUpdates, sliceSize] } });
    toDispose.push(flattenX);
    const flattenTensor = reshape({ inputs: { x: tensor }, backend, attrs: { shape: flattenShape } });
    toDispose.push(flattenTensor);
    const output = tile({
        inputs: { x: flattenTensor },
        backend,
        attrs: { reps: Array(flattenShape.length).fill(1) }
    });
    const program = new ScatterProgram([numUpdates, sliceSize], sliceRank, flattenIndices.shape.length, flattenX.shape.length, strides, flattenShape, tensor.dtype, false);
    const size = util.sizeFromShape([numUpdates, sliceSize]);
    const uniformData = [
        { type: 'int32', data: [sliceRank] },
        { type: 'int32', data: strides },
        { type: 'int32', data: [size] },
    ];
    const res = backend.runWebGPUProgram(program, [flattenX, flattenIndices], flattenTensor.dtype, uniformData, output);
    toDispose.push(res);
    const reshaped = reshape({ inputs: { x: res }, backend, attrs: { shape: tensor.shape } });
    toDispose.forEach(t => backend.disposeData(t.dataId));
    return reshaped;
}
export const tensorScatterUpdateConfig = {
    kernelName: TensorScatterUpdate,
    backendName: 'webgpu',
    kernelFunc: tensorScatterUpdate
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiVGVuc29yU2NhdHRlclVwZGF0ZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvVGVuc29yU2NhdHRlclVwZGF0ZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUF3QyxtQkFBbUIsRUFBdUQsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFHekssT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBRWpELE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUU1QixNQUFNLFVBQVUsbUJBQW1CLENBQUMsSUFJbkM7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsT0FBTyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQzFDLE1BQU0sRUFBRSxHQUFHLEtBQUssQ0FBQztJQUVqQixNQUFNLEVBQUMsU0FBUyxFQUFFLFVBQVUsRUFBRSxTQUFTLEVBQUUsT0FBTyxFQUFFLFVBQVUsRUFBQyxHQUN6RCxZQUFZLENBQUMsZUFBZSxDQUFDLE9BQU8sRUFBRSxPQUFPLEVBQUUsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRWpFLE1BQU0sWUFBWSxHQUFHLENBQUMsVUFBVSxHQUFHLFNBQVMsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUV6RCxJQUFJLFVBQVUsS0FBSyxDQUFDLEVBQUU7UUFDcEIsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO0tBQzVEO0lBRUQsTUFBTSxTQUFTLEdB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QyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIzIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mbywgVGVuc29yU2NhdHRlclVwZGF0ZSwgVGVuc29yU2NhdHRlclVwZGF0ZUF0dHJzLCBUZW5zb3JTY2F0dGVyVXBkYXRlSW5wdXRzLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7U2NhdHRlclByb2dyYW19IGZyb20gJy4uL3NjYXR0ZXJfd2ViZ3B1JztcblxuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL1Jlc2hhcGUnO1xuaW1wb3J0IHt0aWxlfSBmcm9tICcuL1RpbGUnO1xuXG5leHBvcnQgZnVuY3Rpb24gdGVuc29yU2NhdHRlclVwZGF0ZShhcmdzOiB7XG4gIGlucHV0czogVGVuc29yU2NhdHRlclVwZGF0ZUlucHV0cyxcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZCxcbiAgYXR0cnM6IFRlbnNvclNjYXR0ZXJVcGRhdGVBdHRyc1xufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7dGVuc29yLCBpbmRpY2VzLCB1cGRhdGVzfSA9IGlucHV0cztcbiAgY29uc3Qge30gPSBhdHRycztcblxuICBjb25zdCB7c2xpY2VSYW5rLCBudW1VcGRhdGVzLCBzbGljZVNpemUsIHN0cmlkZXMsIG91dHB1dFNpemV9ID1cbiAgICAgIGJhY2tlbmRfdXRpbC5jYWxjdWxhdGVTaGFwZXModXBkYXRlcywgaW5kaWNlcywgdGVuc29yLnNoYXBlKTtcblxuICBjb25zdCBmbGF0dGVuU2hhcGUgPSBbb3V0cHV0U2l6ZSAvIHNsaWNlU2l6ZSwgc2xpY2VTaXplXTtcblxuICBpZiAob3V0cHV0U2l6ZSA9PT0gMCkge1xuICAgIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKHRlbnNvci5zaGFwZSwgaW5kaWNlcy5kdHlwZSk7XG4gIH1cblxuICBjb25zdCB0b0Rpc3Bvc2UgPSBbXTtcblxuICBjb25zdCBmbGF0dGVuSW5kaWNlcyA9IHJlc2hhcGUoXG4gICAgICB7aW5wdXRzOiB7eDogaW5kaWNlc30sIGJhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IFtudW1VcGRhdGVzLCBzbGljZVJhbmtdfX0pO1xuICB0b0Rpc3Bvc2UucHVzaChmbGF0dGVuSW5kaWNlcyk7XG4gIGNvbnN0IGZsYXR0ZW5YID0gcmVzaGFwZShcbiAgICAgIHtpbnB1dHM6IHt4OiB1cGRhdGVzfSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogW251bVVwZGF0ZXMsIHNsaWNlU2l6ZV19fSk7XG4gIHRvRGlzcG9zZS5wdXNoKGZsYXR0ZW5YKTtcbiAgY29uc3QgZmxhdHRlblRlbnNvciA9XG4gICAgICByZXNoYXBlKHtpbnB1dHM6IHt4OiB0ZW5zb3J9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiBmbGF0dGVuU2hhcGV9fSk7XG4gIHRvRGlzcG9zZS5wdXNoKGZsYXR0ZW5UZW5zb3IpO1xuICBjb25zdCBvdXRwdXQgPSB0aWxlKHtcbiAgICBpbnB1dHM6IHt4OiBmbGF0dGVuVGVuc29yfSxcbiAgICBiYWNrZW5kLFxuICAgIGF0dHJzOiB7cmVwczogQXJyYXkoZmxhdHRlblNoYXBlLmxlbmd0aCkuZmlsbCgxKX1cbiAgfSk7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgU2NhdHRlclByb2dyYW0oXG4gICAgICBbbnVtVXBkYXRlcywgc2xpY2VTaXplXSwgc2xpY2VSYW5rLCBmbGF0dGVuSW5kaWNlcy5zaGFwZS5sZW5ndGgsXG4gICAgICBmbGF0dGVuWC5zaGFwZS5sZW5ndGgsIHN0cmlkZXMsIGZsYXR0ZW5TaGFwZSwgdGVuc29yLmR0eXBlLCBmYWxzZSk7XG4gIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoW251bVVwZGF0ZXMsIHNsaWNlU2l6ZV0pO1xuICBjb25zdCB1bmlmb3JtRGF0YSA9IFtcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW3NsaWNlUmFua119LFxuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBzdHJpZGVzfSxcbiAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW3NpemVdfSxcbiAgXTtcbiAgY29uc3QgcmVzID0gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKFxuICAgICAgcHJvZ3JhbSwgW2ZsYXR0ZW5YLCBmbGF0dGVuSW5kaWNlc10sIGZsYXR0ZW5UZW5zb3IuZHR5cGUsIHVuaWZvcm1EYXRhLFxuICAgICAgb3V0cHV0KTtcbiAgdG9EaXNwb3NlLnB1c2gocmVzKTtcblxuICBjb25zdCByZXNoYXBlZCA9XG4gICAgICByZXNoYXBlKHtpbnB1dHM6IHt4OiByZXN9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiB0ZW5zb3Iuc2hhcGV9fSk7XG5cbiAgdG9EaXNwb3NlLmZvckVhY2godCA9PiBiYWNrZW5kLmRpc3Bvc2VEYXRhKHQuZGF0YUlkKSk7XG5cbiAgcmV0dXJuIHJlc2hhcGVkO1xufVxuXG5leHBvcnQgY29uc3QgdGVuc29yU2NhdHRlclVwZGF0ZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBUZW5zb3JTY2F0dGVyVXBkYXRlLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IHRlbnNvclNjYXR0ZXJVcGRhdGUgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19