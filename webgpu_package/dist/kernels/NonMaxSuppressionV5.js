/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { kernel_impls, NonMaxSuppressionV5 } from '@tensorflow/tfjs-core';
export function nonMaxSuppressionV5(args) {
    console.warn('tf.nonMaxSuppression() in webgpu locks the UI thread. ' +
        'Call tf.nonMaxSuppressionAsync() instead');
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
    const boxesVals = backend.readSync(boxes.dataId);
    const scoresVals = backend.readSync(scores.dataId);
    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;
    const { selectedIndices, selectedScores } = kernel_impls.nonMaxSuppressionV5Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
    return [
        backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
        backend.makeTensorInfo([selectedScores.length], 'float32', new Float32Array(selectedScores))
    ];
}
export const nonMaxSuppressionV5Config = {
    kernelName: NonMaxSuppressionV5,
    backendName: 'webgpu',
    kernelFunc: nonMaxSuppressionV5
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTm9uTWF4U3VwcHJlc3Npb25WNS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvTm9uTWF4U3VwcHJlc3Npb25WNS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsWUFBWSxFQUE0QixtQkFBbUIsRUFBa0UsTUFBTSx1QkFBdUIsQ0FBQztBQUtuSyxNQUFNLFVBQVUsbUJBQW1CLENBQUMsSUFJbkM7SUFDQyxPQUFPLENBQUMsSUFBSSxDQUNSLHdEQUF3RDtRQUN4RCwwQ0FBMEMsQ0FBQyxDQUFDO0lBRWhELE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsS0FBSyxFQUFFLE1BQU0sRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMvQixNQUFNLEVBQUMsYUFBYSxFQUFFLFlBQVksRUFBRSxjQUFjLEVBQUUsWUFBWSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRTFFLE1BQU0sU0FBUyxHQUFHLE9BQU8sQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBZSxDQUFDO0lBQy9ELE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxRQUFRLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBZSxDQUFDO0lBRWpFLE1BQU0sZ0JBQWdCLEdBQUcsYUFBYSxDQUFDO0lBQ3ZDLE1BQU0sZUFBZSxHQUFHLFlBQVksQ0FBQztJQUNyQyxNQUFNLGlCQUFpQixHQUFHLGNBQWMsQ0FBQztJQUN6QyxNQUFNLGVBQWUsR0FBRyxZQUFZLENBQUM7SUFFckMsTUFBTSxFQUFDLGVBQWUsRUFBRSxjQUFjLEVBQUMsR0FDbkMsWUFBWSxDQUFDLHVCQUF1QixDQUNoQyxTQUFTLEVBQUUsVUFBVSxFQUFFLGdCQUFnQixFQUFFLGVBQWUsRUFDeEQsaUJBQWlCLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFFNUMsT0FBTztRQUNMLE9BQU8sQ0FBQyxjQUFjLENBQ2xCLENBQUMsZUFBZSxDQUFDLE1BQU0sQ0FBQyxFQUFFLE9BQU8sRUFBRSxJQUFJLFVBQVUsQ0FBQyxlQUFlLENBQUMsQ0FBQztRQUN2RSxPQUFPLENBQUMsY0FBYyxDQUNsQixDQUFDLGNBQWMsQ0FBQyxNQUFNLENBQUMsRUFBRSxTQUFTLEVBQUUsSUFBSSxZQUFZLENBQUMsY0FBYyxDQUFDLENBQUM7S0FDMUUsQ0FBQztBQUNKLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSx5QkFBeUIsR0FBaUI7SUFDckQsVUFBVSxFQUFFLG1CQUFtQjtJQUMvQixXQUFXLEVBQUUsUUFBUTtJQUNyQixVQUFVLEVBQUUsbUJBQTRDO0NBQ3pELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge2tlcm5lbF9pbXBscywgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBOb25NYXhTdXBwcmVzc2lvblY1LCBOb25NYXhTdXBwcmVzc2lvblY1QXR0cnMsIE5vbk1heFN1cHByZXNzaW9uVjVJbnB1dHMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuZXhwb3J0IHR5cGUgVHlwZWRBcnJheSA9IEZsb2F0MzJBcnJheXxJbnQzMkFycmF5fFVpbnQ4QXJyYXk7XG5cbmV4cG9ydCBmdW5jdGlvbiBub25NYXhTdXBwcmVzc2lvblY1KGFyZ3M6IHtcbiAgaW5wdXRzOiBOb25NYXhTdXBwcmVzc2lvblY1SW5wdXRzLFxuICBiYWNrZW5kOiBXZWJHUFVCYWNrZW5kLFxuICBhdHRyczogTm9uTWF4U3VwcHJlc3Npb25WNUF0dHJzXG59KTogW1RlbnNvckluZm8sIFRlbnNvckluZm9dIHtcbiAgY29uc29sZS53YXJuKFxuICAgICAgJ3RmLm5vbk1heFN1cHByZXNzaW9uKCkgaW4gd2ViZ3B1IGxvY2tzIHRoZSBVSSB0aHJlYWQuICcgK1xuICAgICAgJ0NhbGwgdGYubm9uTWF4U3VwcHJlc3Npb25Bc3luYygpIGluc3RlYWQnKTtcblxuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7Ym94ZXMsIHNjb3Jlc30gPSBpbnB1dHM7XG4gIGNvbnN0IHttYXhPdXRwdXRTaXplLCBpb3VUaHJlc2hvbGQsIHNjb3JlVGhyZXNob2xkLCBzb2Z0Tm1zU2lnbWF9ID0gYXR0cnM7XG5cbiAgY29uc3QgYm94ZXNWYWxzID0gYmFja2VuZC5yZWFkU3luYyhib3hlcy5kYXRhSWQpIGFzIFR5cGVkQXJyYXk7XG4gIGNvbnN0IHNjb3Jlc1ZhbHMgPSBiYWNrZW5kLnJlYWRTeW5jKHNjb3Jlcy5kYXRhSWQpIGFzIFR5cGVkQXJyYXk7XG5cbiAgY29uc3QgbWF4T3V0cHV0U2l6ZVZhbCA9IG1heE91dHB1dFNpemU7XG4gIGNvbnN0IGlvdVRocmVzaG9sZFZhbCA9IGlvdVRocmVzaG9sZDtcbiAgY29uc3Qgc2NvcmVUaHJlc2hvbGRWYWwgPSBzY29yZVRocmVzaG9sZDtcbiAgY29uc3Qgc29mdE5tc1NpZ21hVmFsID0gc29mdE5tc1NpZ21hO1xuXG4gIGNvbnN0IHtzZWxlY3RlZEluZGljZXMsIHNlbGVjdGVkU2NvcmVzfSA9XG4gICAgICBrZXJuZWxfaW1wbHMubm9uTWF4U3VwcHJlc3Npb25WNUltcGwoXG4gICAgICAgICAgYm94ZXNWYWxzLCBzY29yZXNWYWxzLCBtYXhPdXRwdXRTaXplVmFsLCBpb3VUaHJlc2hvbGRWYWwsXG4gICAgICAgICAgc2NvcmVUaHJlc2hvbGRWYWwsIHNvZnRObXNTaWdtYVZhbCk7XG5cbiAgcmV0dXJuIFtcbiAgICBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKFxuICAgICAgICBbc2VsZWN0ZWRJbmRpY2VzLmxlbmd0aF0sICdpbnQzMicsIG5ldyBJbnQzMkFycmF5KHNlbGVjdGVkSW5kaWNlcykpLFxuICAgIGJhY2tlbmQubWFrZVRlbnNvckluZm8oXG4gICAgICAgIFtzZWxlY3RlZFNjb3Jlcy5sZW5ndGhdLCAnZmxvYXQzMicsIG5ldyBGbG9hdDMyQXJyYXkoc2VsZWN0ZWRTY29yZXMpKVxuICBdO1xufVxuXG5leHBvcnQgY29uc3Qgbm9uTWF4U3VwcHJlc3Npb25WNUNvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBOb25NYXhTdXBwcmVzc2lvblY1LFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IG5vbk1heFN1cHByZXNzaW9uVjUgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19