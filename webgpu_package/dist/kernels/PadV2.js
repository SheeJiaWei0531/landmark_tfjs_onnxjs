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
import { PadV2, util } from '@tensorflow/tfjs-core';
import { identity } from './Identity';
import { PadProgram } from '../pad_webgpu';
import { fill } from './Fill';
export const padV2 = (args) => {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { paddings, constantValue } = attrs;
    if (paddings.every(p => util.arraysEqual(p, [0, 0]))) {
        return identity({ inputs: { x }, backend });
    }
    if (util.sizeFromShape(x.shape) === 0) {
        // Short-circuit the computation, since x doesn't have value, only
        // the shape is used to compute output shape to pad.
        const outputShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
        return fill({
            backend,
            attrs: { shape: outputShape, value: constantValue, dtype: x.dtype }
        });
    }
    const uniformData = [{ type: 'float32', data: [constantValue] }];
    paddings.map(p => uniformData.push({ type: 'int32', data: [p[0], p[1]] }));
    const program = new PadProgram(x.shape, paddings);
    return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
};
export const padV2Config = {
    kernelName: PadV2,
    backendName: 'webgpu',
    kernelFunc: padV2
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUGFkVjIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL1BhZFYyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBMkIsS0FBSyxFQUF1QyxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUdqSCxPQUFPLEVBQUMsUUFBUSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxVQUFVLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDekMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUU1QixNQUFNLENBQUMsTUFBTSxLQUFLLEdBQ2QsQ0FBQyxJQUV5QixFQUFjLEVBQUU7SUFDeEMsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxDQUFDLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDbkIsTUFBTSxFQUFDLFFBQVEsRUFBRSxhQUFhLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFDeEMsSUFBSSxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFO1FBQ3BELE9BQU8sUUFBUSxDQUFDLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFDLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztLQUN6QztJQUNELElBQUksSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxFQUFFO1FBQ3JDLGtFQUFrRTtRQUNsRSxvREFBb0Q7UUFDcEQsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDLEdBQUcsQ0FDNUIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FDTCxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsZUFBZSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxDQUFDO1FBQ2pFLE9BQU8sSUFBSSxDQUFDO1lBQ1YsT0FBTztZQUNQLEtBQUssRUFBRSxFQUFDLEtBQUssRUFBRSxXQUFXLEVBQUUsS0FBSyxFQUFFLGFBQWEsRUFBRSxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBQztTQUNsRSxDQUFDLENBQUM7S0FDSjtJQUNELE1BQU0sV0FBVyxHQUFHLENBQUMsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLGFBQWEsQ0FBQyxFQUFDLENBQUMsQ0FBQztJQUMvRCxRQUFRLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pFLE1BQU0sT0FBTyxHQUFHLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbEQsT0FBTyxPQUFPLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztBQUN0RSxDQUFDLENBQUM7QUFFTixNQUFNLENBQUMsTUFBTSxXQUFXLEdBQWlCO0lBQ3ZDLFVBQVUsRUFBRSxLQUFLO0lBQ2pCLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxLQUE4QjtDQUMzQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgUGFkVjIsIFBhZFYyQXR0cnMsIFBhZFYySW5wdXRzLCBUZW5zb3JJbmZvLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7aWRlbnRpdHl9IGZyb20gJy4vSWRlbnRpdHknO1xuaW1wb3J0IHtQYWRQcm9ncmFtfSBmcm9tICcuLi9wYWRfd2ViZ3B1JztcbmltcG9ydCB7ZmlsbH0gZnJvbSAnLi9GaWxsJztcblxuZXhwb3J0IGNvbnN0IHBhZFYyID1cbiAgICAoYXJnczoge2lucHV0czogUGFkVjJJbnB1dHMsXG4gICAgICAgICAgICBiYWNrZW5kOiBXZWJHUFVCYWNrZW5kLFxuICAgICAgICAgICAgYXR0cnM6IFBhZFYyQXR0cnN9KTogVGVuc29ySW5mbyA9PiB7XG4gICAgICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICAgICAgY29uc3Qge3h9ID0gaW5wdXRzO1xuICAgICAgY29uc3Qge3BhZGRpbmdzLCBjb25zdGFudFZhbHVlfSA9IGF0dHJzO1xuICAgICAgaWYgKHBhZGRpbmdzLmV2ZXJ5KHAgPT4gdXRpbC5hcnJheXNFcXVhbChwLCBbMCwgMF0pKSkge1xuICAgICAgICByZXR1cm4gaWRlbnRpdHkoe2lucHV0czoge3h9LCBiYWNrZW5kfSk7XG4gICAgICB9XG4gICAgICBpZiAodXRpbC5zaXplRnJvbVNoYXBlKHguc2hhcGUpID09PSAwKSB7XG4gICAgICAgIC8vIFNob3J0LWNpcmN1aXQgdGhlIGNvbXB1dGF0aW9uLCBzaW5jZSB4IGRvZXNuJ3QgaGF2ZSB2YWx1ZSwgb25seVxuICAgICAgICAvLyB0aGUgc2hhcGUgaXMgdXNlZCB0byBjb21wdXRlIG91dHB1dCBzaGFwZSB0byBwYWQuXG4gICAgICAgIGNvbnN0IG91dHB1dFNoYXBlID0gcGFkZGluZ3MubWFwKFxuICAgICAgICAgICAgKHAsIGkpID0+XG4gICAgICAgICAgICAgICAgcFswXSAvKiBiZWZvcmVQYWQgKi8gKyB4LnNoYXBlW2ldICsgcFsxXSAvKiBhZnRlclBhZCAqLyk7XG4gICAgICAgIHJldHVybiBmaWxsKHtcbiAgICAgICAgICBiYWNrZW5kLFxuICAgICAgICAgIGF0dHJzOiB7c2hhcGU6IG91dHB1dFNoYXBlLCB2YWx1ZTogY29uc3RhbnRWYWx1ZSwgZHR5cGU6IHguZHR5cGV9XG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgICAgY29uc3QgdW5pZm9ybURhdGEgPSBbe3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW2NvbnN0YW50VmFsdWVdfV07XG4gICAgICBwYWRkaW5ncy5tYXAocCA9PiB1bmlmb3JtRGF0YS5wdXNoKHt0eXBlOiAnaW50MzInLCBkYXRhOiBbcFswXSwgcFsxXV19KSk7XG4gICAgICBjb25zdCBwcm9ncmFtID0gbmV3IFBhZFByb2dyYW0oeC5zaGFwZSwgcGFkZGluZ3MpO1xuICAgICAgcmV0dXJuIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBbeF0sIHguZHR5cGUsIHVuaWZvcm1EYXRhKTtcbiAgICB9O1xuXG5leHBvcnQgY29uc3QgcGFkVjJDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogUGFkVjIsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ3B1JyxcbiAga2VybmVsRnVuYzogcGFkVjIgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19