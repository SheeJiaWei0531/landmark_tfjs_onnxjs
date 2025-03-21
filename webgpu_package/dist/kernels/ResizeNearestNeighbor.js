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
import { ResizeNearestNeighbor } from '@tensorflow/tfjs-core';
import { ResizeNearestNeighborProgram } from '../resize_nearest_neighbor_webgpu';
export function resizeNearestNeighbor(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    const [newHeight, newWidth] = size;
    const adjustHeight = alignCorners && newHeight > 1 ? 1.0 : 0.0;
    const adjustWidth = alignCorners && newWidth > 1 ? 1.0 : 0.0;
    // When align corners is false, we rounds the value with floor.
    const roundBase = alignCorners ? 0.5 : 0.0;
    const uniformData = [
        { type: 'float32', data: [adjustHeight, adjustWidth] },
        { type: 'float32', data: [roundBase] }
    ];
    const program = new ResizeNearestNeighborProgram(images.shape, newHeight, newWidth, halfPixelCenters);
    return backend.runWebGPUProgram(program, [images], images.dtype, uniformData);
}
export const resizeNearestNeighborConfig = {
    kernelName: ResizeNearestNeighbor,
    backendName: 'webgpu',
    kernelFunc: resizeNearestNeighbor
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiUmVzaXplTmVhcmVzdE5laWdoYm9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdwdS9zcmMva2VybmVscy9SZXNpemVOZWFyZXN0TmVpZ2hib3IudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixxQkFBcUIsRUFBc0UsTUFBTSx1QkFBdUIsQ0FBQztBQUczSixPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxtQ0FBbUMsQ0FBQztBQUUvRSxNQUFNLFVBQVUscUJBQXFCLENBQUMsSUFJckM7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLE1BQU0sRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUN4QixNQUFNLEVBQUMsWUFBWSxFQUFFLGdCQUFnQixFQUFFLElBQUksRUFBQyxHQUFHLEtBQUssQ0FBQztJQUVyRCxNQUFNLENBQUMsU0FBUyxFQUFFLFFBQVEsQ0FBQyxHQUFHLElBQUksQ0FBQztJQUNuQyxNQUFNLFlBQVksR0FBRyxZQUFZLElBQUksU0FBUyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUM7SUFDL0QsTUFBTSxXQUFXLEdBQUcsWUFBWSxJQUFJLFFBQVEsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO0lBQzdELCtEQUErRDtJQUMvRCxNQUFNLFNBQVMsR0FBRyxZQUFZLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO0lBQzNDLE1BQU0sV0FBVyxHQUFHO1FBQ2xCLEVBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxZQUFZLEVBQUUsV0FBVyxDQUFDLEVBQUM7UUFDcEQsRUFBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLElBQUksRUFBRSxDQUFDLFNBQVMsQ0FBQyxFQUFDO0tBQ3JDLENBQUM7SUFFRixNQUFNLE9BQU8sR0FBRyxJQUFJLDRCQUE0QixDQUM1QyxNQUFNLENBQUMsS0FBeUMsRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUNyRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBQ3RCLE9BQU8sT0FBTyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxDQUFDLE1BQU0sQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDaEYsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLDJCQUEyQixHQUFpQjtJQUN2RCxVQUFVLEVBQUUscUJBQXFCO0lBQ2pDLFdBQVcsRUFBRSxRQUFRO0lBQ3JCLFVBQVUsRUFBRSxxQkFBOEM7Q0FDM0QsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFJlc2l6ZU5lYXJlc3ROZWlnaGJvciwgUmVzaXplTmVhcmVzdE5laWdoYm9yQXR0cnMsIFJlc2l6ZU5lYXJlc3ROZWlnaGJvcklucHV0cywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5pbXBvcnQge1Jlc2l6ZU5lYXJlc3ROZWlnaGJvclByb2dyYW19IGZyb20gJy4uL3Jlc2l6ZV9uZWFyZXN0X25laWdoYm9yX3dlYmdwdSc7XG5cbmV4cG9ydCBmdW5jdGlvbiByZXNpemVOZWFyZXN0TmVpZ2hib3IoYXJnczoge1xuICBpbnB1dHM6IFJlc2l6ZU5lYXJlc3ROZWlnaGJvcklucHV0cyxcbiAgYmFja2VuZDogV2ViR1BVQmFja2VuZCxcbiAgYXR0cnM6IFJlc2l6ZU5lYXJlc3ROZWlnaGJvckF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtpbWFnZXN9ID0gaW5wdXRzO1xuICBjb25zdCB7YWxpZ25Db3JuZXJzLCBoYWxmUGl4ZWxDZW50ZXJzLCBzaXplfSA9IGF0dHJzO1xuXG4gIGNvbnN0IFtuZXdIZWlnaHQsIG5ld1dpZHRoXSA9IHNpemU7XG4gIGNvbnN0IGFkanVzdEhlaWdodCA9IGFsaWduQ29ybmVycyAmJiBuZXdIZWlnaHQgPiAxID8gMS4wIDogMC4wO1xuICBjb25zdCBhZGp1c3RXaWR0aCA9IGFsaWduQ29ybmVycyAmJiBuZXdXaWR0aCA+IDEgPyAxLjAgOiAwLjA7XG4gIC8vIFdoZW4gYWxpZ24gY29ybmVycyBpcyBmYWxzZSwgd2Ugcm91bmRzIHRoZSB2YWx1ZSB3aXRoIGZsb29yLlxuICBjb25zdCByb3VuZEJhc2UgPSBhbGlnbkNvcm5lcnMgPyAwLjUgOiAwLjA7XG4gIGNvbnN0IHVuaWZvcm1EYXRhID0gW1xuICAgIHt0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFthZGp1c3RIZWlnaHQsIGFkanVzdFdpZHRoXX0sXG4gICAge3R5cGU6ICdmbG9hdDMyJywgZGF0YTogW3JvdW5kQmFzZV19XG4gIF07XG5cbiAgY29uc3QgcHJvZ3JhbSA9IG5ldyBSZXNpemVOZWFyZXN0TmVpZ2hib3JQcm9ncmFtKFxuICAgICAgaW1hZ2VzLnNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBuZXdIZWlnaHQsIG5ld1dpZHRoLFxuICAgICAgaGFsZlBpeGVsQ2VudGVycyk7XG4gIHJldHVybiBiYWNrZW5kLnJ1bldlYkdQVVByb2dyYW0ocHJvZ3JhbSwgW2ltYWdlc10sIGltYWdlcy5kdHlwZSwgdW5pZm9ybURhdGEpO1xufVxuXG5leHBvcnQgY29uc3QgcmVzaXplTmVhcmVzdE5laWdoYm9yQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFJlc2l6ZU5lYXJlc3ROZWlnaGJvcixcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiByZXNpemVOZWFyZXN0TmVpZ2hib3IgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19