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
import { backend_util, FusedDepthwiseConv2D, util } from '@tensorflow/tfjs-core';
import { DepthwiseConv2DVec4Program } from '../depthwise_conv2d_vec4_webgpu';
import { DepthwiseConv2DProgram } from '../depthwise_conv2d_webgpu';
export function fusedDepthwiseConv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    util.assert(backend_util.eitherStridesOrDilationsAreOne(strides, $dilations), () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
        `1. Got strides ${strides} and dilations '${$dilations}'`);
    const convInfo = backend_util.computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
    const programInputs = [x, filter];
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasBias) {
        programInputs.push(bias);
    }
    if (hasPreluActivationWeights) {
        programInputs.push(preluActivationWeights);
    }
    const dimensions = [
        { type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left] },
        { type: 'int32', data: [convInfo.inHeight, convInfo.inWidth] },
    ];
    let program;
    if (convInfo.outHeight > 4 && convInfo.outWidth > 4 &&
        convInfo.strideWidth <= 2 &&
        convInfo.inChannels === convInfo.outChannels &&
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
        convInfo.inChannels % 4 === 0) {
        program = new DepthwiseConv2DVec4Program(convInfo, hasBias, activation, hasPreluActivationWeights);
        dimensions.push({ type: 'int32', data: [program.virtualWidth] });
    }
    else {
        program = new DepthwiseConv2DProgram(convInfo, hasBias, activation, hasPreluActivationWeights);
        dimensions.push({ type: 'int32', data: [convInfo.filterHeight] }, { type: 'int32', data: [convInfo.filterWidth] }, { type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth] }, {
            type: 'int32',
            data: [convInfo.dilationHeight, convInfo.dilationWidth]
        });
    }
    if (activation === 'leakyrelu') {
        dimensions.push({ type: 'float32', data: [leakyreluAlpha] });
        program.uniforms += ' alpha : f32,';
    }
    const result = backend.runWebGPUProgram(program, programInputs, 'float32', dimensions);
    return result;
}
export const fusedDepthwiseConv2DConfig = {
    kernelName: FusedDepthwiseConv2D,
    backendName: 'webgpu',
    kernelFunc: fusedDepthwiseConv2D,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRnVzZWREZXB0aHdpc2VDb252MkQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ3B1L3NyYy9rZXJuZWxzL0Z1c2VkRGVwdGh3aXNlQ29udjJELnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQUUsb0JBQW9CLEVBQStGLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRzVLLE9BQU8sRUFBQywwQkFBMEIsRUFBQyxNQUFNLGlDQUFpQyxDQUFDO0FBQzNFLE9BQU8sRUFBQyxzQkFBc0IsRUFBQyxNQUFNLDRCQUE0QixDQUFDO0FBRWxFLE1BQU0sVUFBVSxvQkFBb0IsQ0FBQyxJQUlwQztJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsc0JBQXNCLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDekQsTUFBTSxFQUFDLE9BQU8sRUFBRSxHQUFHLEVBQUUsU0FBUyxFQUFFLGVBQWUsRUFBRSxVQUFVLEVBQUUsY0FBYyxFQUFDLEdBQ3hFLEtBQUssQ0FBQztJQUVWLElBQUksVUFBVSxHQUFHLFNBQVMsQ0FBQztJQUMzQixJQUFJLFVBQVUsSUFBSSxJQUFJLEVBQUU7UUFDdEIsVUFBVSxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0tBQ3JCO0lBRUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxZQUFZLENBQUMsOEJBQThCLENBQUMsT0FBTyxFQUFFLFVBQVUsQ0FBQyxFQUNoRSxHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsa0JBQWtCLE9BQU8sbUJBQW1CLFVBQVUsR0FBRyxDQUFDLENBQUM7SUFFbkUsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLGlCQUFp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QixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sMEJBQTBCLEdBQWlCO0lBQ3RELFVBQVUsRUFBRSxvQkFBb0I7SUFDaEMsV0FBVyxFQUFFLFFBQVE7SUFDckIsVUFBVSxFQUFFLG9CQUE2QztDQUMxRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgRnVzZWREZXB0aHdpc2VDb252MkQsIEZ1c2VkRGVwdGh3aXNlQ29udjJEQXR0cnMsIEZ1c2VkRGVwdGh3aXNlQ29udjJESW5wdXRzLCBLZXJuZWxDb25maWcsIEtlcm5lbEZ1bmMsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7V2ViR1BVQmFja2VuZH0gZnJvbSAnLi4vYmFja2VuZF93ZWJncHUnO1xuaW1wb3J0IHtEZXB0aHdpc2VDb252MkRWZWM0UHJvZ3JhbX0gZnJvbSAnLi4vZGVwdGh3aXNlX2NvbnYyZF92ZWM0X3dlYmdwdSc7XG5pbXBvcnQge0RlcHRod2lzZUNvbnYyRFByb2dyYW19IGZyb20gJy4uL2RlcHRod2lzZV9jb252MmRfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGZ1c2VkRGVwdGh3aXNlQ29udjJEKGFyZ3M6IHtcbiAgaW5wdXRzOiBGdXNlZERlcHRod2lzZUNvbnYyRElucHV0cyxcbiAgYXR0cnM6IEZ1c2VkRGVwdGh3aXNlQ29udjJEQXR0cnMsXG4gIGJhY2tlbmQ6IFdlYkdQVUJhY2tlbmRcbn0pIHtcbiAgY29uc3Qge2lucHV0cywgYmFja2VuZCwgYXR0cnN9ID0gYXJncztcbiAgY29uc3Qge3gsIGZpbHRlciwgYmlhcywgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0c30gPSBpbnB1dHM7XG4gIGNvbnN0IHtzdHJpZGVzLCBwYWQsIGRpbGF0aW9ucywgZGltUm91bmRpbmdNb2RlLCBhY3RpdmF0aW9uLCBsZWFreXJlbHVBbHBoYX0gPVxuICAgICAgYXR0cnM7XG5cbiAgbGV0ICRkaWxhdGlvbnMgPSBkaWxhdGlvbnM7XG4gIGlmICgkZGlsYXRpb25zID09IG51bGwpIHtcbiAgICAkZGlsYXRpb25zID0gWzEsIDFdO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICBiYWNrZW5kX3V0aWwuZWl0aGVyU3RyaWRlc09yRGlsYXRpb25zQXJlT25lKHN0cmlkZXMsICRkaWxhdGlvbnMpLFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIGRlcHRod2lzZUNvbnYyZDogRWl0aGVyIHN0cmlkZXMgb3IgZGlsYXRpb25zIG11c3QgYmUgJyArXG4gICAgICAgICAgYDEuIEdvdCBzdHJpZGVzICR7c3RyaWRlc30gYW5kIGRpbGF0aW9ucyAnJHskZGlsYXRpb25zfSdgKTtcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlQ29udjJESW5mbyhcbiAgICAgIHguc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICBmaWx0ZXIuc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHN0cmlkZXMsICRkaWxhdGlvbnMsXG4gICAgICBwYWQsIGRpbVJvdW5kaW5nTW9kZSwgdHJ1ZSAvKiBkZXB0aHdpc2UgKi8pO1xuXG4gIGNvbnN0IHByb2dyYW1JbnB1dHM6IFRlbnNvckluZm9bXSA9IFt4LCBmaWx0ZXJdO1xuXG4gIGNvbnN0IGhhc0JpYXMgPSBiaWFzICE9IG51bGw7XG4gIGNvbnN0IGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMgPSBwcmVsdUFjdGl2YXRpb25XZWlnaHRzICE9IG51bGw7XG5cbiAgaWYgKGhhc0JpYXMpIHtcbiAgICBwcm9ncmFtSW5wdXRzLnB1c2goYmlhcyk7XG4gIH1cbiAgaWYgKGhhc1ByZWx1QWN0aXZhdGlvbldlaWdodHMpIHtcbiAgICBwcm9ncmFtSW5wdXRzLnB1c2gocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyk7XG4gIH1cblxuICBjb25zdCBkaW1lbnNpb25zID0gW1xuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbY29udkluZm8ucGFkSW5mby50b3AsIGNvbnZJbmZvLnBhZEluZm8ubGVmdF19LFxuICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbY29udkluZm8uaW5IZWlnaHQsIGNvbnZJbmZvLmluV2lkdGhdfSxcbiAgXTtcblxuICBsZXQgcHJvZ3JhbTogRGVwdGh3aXNlQ29udjJEUHJvZ3JhbXxEZXB0aHdpc2VDb252MkRWZWM0UHJvZ3JhbTtcbiAgaWYgKGNvbnZJbmZvLm91dEhlaWdodCA+IDQgJiYgY29udkluZm8ub3V0V2lkdGggPiA0ICYmXG4gICAgICBjb252SW5mby5zdHJpZGVXaWR0aCA8PSAyICYmXG4gICAgICBjb252SW5mby5pbkNoYW5uZWxzID09PSBjb252SW5mby5vdXRDaGFubmVscyAmJlxuICAgICAgY29udkluZm8uZGlsYXRpb25IZWlnaHQgPT09IDEgJiYgY29udkluZm8uZGlsYXRpb25XaWR0aCA9PT0gMSAmJlxuICAgICAgY29udkluZm8uaW5DaGFubmVscyAlIDQgPT09IDApIHtcbiAgICBwcm9ncmFtID0gbmV3IERlcHRod2lzZUNvbnYyRFZlYzRQcm9ncmFtKFxuICAgICAgICBjb252SW5mbywgaGFzQmlhcywgYWN0aXZhdGlvbiwgaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyk7XG4gICAgZGltZW5zaW9ucy5wdXNoKHt0eXBlOiAnaW50MzInLCBkYXRhOiBbcHJvZ3JhbS52aXJ0dWFsV2lkdGhdfSk7XG4gIH0gZWxzZSB7XG4gICAgcHJvZ3JhbSA9IG5ldyBEZXB0aHdpc2VDb252MkRQcm9ncmFtKFxuICAgICAgICBjb252SW5mbywgaGFzQmlhcywgYWN0aXZhdGlvbiwgaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyk7XG4gICAgZGltZW5zaW9ucy5wdXNoKFxuICAgICAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2NvbnZJbmZvLmZpbHRlckhlaWdodF19LFxuICAgICAgICB7dHlwZTogJ2ludDMyJywgZGF0YTogW2NvbnZJbmZvLmZpbHRlcldpZHRoXX0sXG4gICAgICAgIHt0eXBlOiAnaW50MzInLCBkYXRhOiBbY29udkluZm8uc3RyaWRlSGVpZ2h0LCBjb252SW5mby5zdHJpZGVXaWR0aF19LCB7XG4gICAgICAgICAgdHlwZTogJ2ludDMyJyxcbiAgICAgICAgICBkYXRhOiBbY29udkluZm8uZGlsYXRpb25IZWlnaHQsIGNvbnZJbmZvLmRpbGF0aW9uV2lkdGhdXG4gICAgICAgIH0pO1xuICB9XG4gIGlmIChhY3RpdmF0aW9uID09PSAnbGVha3lyZWx1Jykge1xuICAgIGRpbWVuc2lvbnMucHVzaCh7dHlwZTogJ2Zsb2F0MzInLCBkYXRhOiBbbGVha3lyZWx1QWxwaGFdfSk7XG4gICAgcHJvZ3JhbS51bmlmb3JtcyArPSAnIGFscGhhIDogZjMyLCc7XG4gIH1cbiAgY29uc3QgcmVzdWx0ID1cbiAgICAgIGJhY2tlbmQucnVuV2ViR1BVUHJvZ3JhbShwcm9ncmFtLCBwcm9ncmFtSW5wdXRzLCAnZmxvYXQzMicsIGRpbWVuc2lvbnMpO1xuXG4gIHJldHVybiByZXN1bHQ7XG59XG5cbmV4cG9ydCBjb25zdCBmdXNlZERlcHRod2lzZUNvbnYyRENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBGdXNlZERlcHRod2lzZUNvbnYyRCxcbiAgYmFja2VuZE5hbWU6ICd3ZWJncHUnLFxuICBrZXJuZWxGdW5jOiBmdXNlZERlcHRod2lzZUNvbnYyRCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmMsXG59O1xuIl19