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
import { CropAndResize } from '@tensorflow/tfjs-core';
import { CropAndResizeProgram } from '../crop_and_resize_webgpu';
export const cropAndResize = (args) => {
    const { inputs, backend, attrs } = args;
    const { image, boxes, boxInd } = inputs;
    const { cropSize, method, extrapolationValue } = attrs;
    const program = new CropAndResizeProgram(image.shape[3], boxes.shape, cropSize, method);
    const uniformData = [{ type: 'float32', data: [extrapolationValue] }];
    return backend.runWebGPUProgram(program, [image, boxes, boxInd], 'float32', uniformData);
};
export const cropAndResizeConfig = {
    kernelName: CropAndResize,
    backendName: 'webgpu',
    kernelFunc: cropAndResize
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ3JvcEFuZFJlc2l6ZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvQ3JvcEFuZFJlc2l6ZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsYUFBYSxFQUFnRixNQUFNLHVCQUF1QixDQUFDO0FBR25JLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLDJCQUEyQixDQUFDO0FBRS9ELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBRyxDQUFDLElBSTdCLEVBQWMsRUFBRTtJQUNmLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDdEMsTUFBTSxFQUFDLFFBQVEsRUFBRSxNQUFNLEVBQUUsa0JBQWtCLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFckQsTUFBTSxPQUFPLEdBQUcsSUFBSSxvQkFBb0IsQ0FDcEMsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsS0FBeUIsRUFBRSxRQUFRLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDdkUsTUFBTSxXQUFXLEdBQUcsQ0FBQyxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLENBQUMsa0JBQWtCLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDcEUsT0FBTyxPQUFPLENBQUMsZ0JBQWdCLENBQzNCLE9BQU8sRUFBRSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLEVBQUUsU0FBUyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQy9ELENBQUMsQ0FBQztBQUVGLE1BQU0sQ0FBQyxNQUFNLG1CQUFtQixHQUFpQjtJQUMvQyxVQUFVLEVBQUUsYUFBYTtJQUN6QixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsYUFBc0M7Q0FDbkQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtDcm9wQW5kUmVzaXplLCBDcm9wQW5kUmVzaXplQXR0cnMsIENyb3BBbmRSZXNpemVJbnB1dHMsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtXZWJHUFVCYWNrZW5kfSBmcm9tICcuLi9iYWNrZW5kX3dlYmdwdSc7XG5pbXBvcnQge0Nyb3BBbmRSZXNpemVQcm9ncmFtfSBmcm9tICcuLi9jcm9wX2FuZF9yZXNpemVfd2ViZ3B1JztcblxuZXhwb3J0IGNvbnN0IGNyb3BBbmRSZXNpemUgPSAoYXJnczoge1xuICBpbnB1dHM6IENyb3BBbmRSZXNpemVJbnB1dHMsXG4gIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmQsXG4gIGF0dHJzOiBDcm9wQW5kUmVzaXplQXR0cnNcbn0pOiBUZW5zb3JJbmZvID0+IHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge2ltYWdlLCBib3hlcywgYm94SW5kfSA9IGlucHV0cztcbiAgY29uc3Qge2Nyb3BTaXplLCBtZXRob2QsIGV4dHJhcG9sYXRpb25WYWx1ZX0gPSBhdHRycztcblxuICBjb25zdCBwcm9ncmFtID0gbmV3IENyb3BBbmRSZXNpemVQcm9ncmFtKFxuICAgICAgaW1hZ2Uuc2hhcGVbM10sIGJveGVzLnNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIGNyb3BTaXplLCBtZXRob2QpO1xuICBjb25zdCB1bmlmb3JtRGF0YSA9IFt7dHlwZTogJ2Zsb2F0MzInLCBkYXRhOiBbZXh0cmFwb2xhdGlvblZhbHVlXX1dO1xuICByZXR1cm4gYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKFxuICAgICAgcHJvZ3JhbSwgW2ltYWdlLCBib3hlcywgYm94SW5kXSwgJ2Zsb2F0MzInLCB1bmlmb3JtRGF0YSk7XG59O1xuXG5leHBvcnQgY29uc3QgY3JvcEFuZFJlc2l6ZUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBDcm9wQW5kUmVzaXplLFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGNyb3BBbmRSZXNpemUgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19