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
import { backend_util, MaxPool3DGrad } from '@tensorflow/tfjs-core';
let wasmMaxPool3DGrad;
function setup(backend) {
    wasmMaxPool3DGrad = backend.wasm.cwrap('MaxPool3DGrad', null, [
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number',
        'number', // padLeft
    ]);
}
export function maxPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = backend_util.computePool3DInfo(input.shape, filterSize, strides, /*dilations=*/ 1, pad, dimRoundingMode);
    const dx = backend.makeOutput(input.shape, input.dtype);
    wasmMaxPool3DGrad(backend.dataIdMap.get(input.dataId).id, backend.dataIdMap.get(dy.dataId).id, backend.dataIdMap.get(dx.dataId).id, convInfo.batchSize, 
    // Since Pool3D ops (MaxPool3D and MaxPool3D) support 3D filter only, in
    // channels should always equal to out channels.
    /*channelSize=*/ convInfo.inChannels, convInfo.inDepth, convInfo.inHeight, convInfo.inWidth, convInfo.outDepth, convInfo.outHeight, convInfo.outWidth, convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth, convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth, convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left);
    return dx;
}
export const maxPool3DGradConfig = {
    kernelName: MaxPool3DGrad,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: maxPool3DGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWF4UG9vbDNER3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13YXNtL3NyYy9rZXJuZWxzL01heFBvb2wzREdyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBNEIsYUFBYSxFQUFzRCxNQUFNLHVCQUF1QixDQUFDO0FBSWpKLElBQUksaUJBTzBELENBQUM7QUFFL0QsU0FBUyxLQUFLLENBQUMsT0FBb0I7SUFDakMsaUJBQWlCLEdBQUcsT0FBTyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsZUFBZSxFQUFFLElBQUksRUFBRTtRQUM1RCxRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUSxFQUFHLFVBQVU7S0FDdEIsQ0FBQyxDQUFDO0FBQ0wsQ0FBQztBQUVELE1BQU0sVUFBVSxhQUFhLENBQUMsSUFJN0I7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLEVBQUUsRUFBRSxLQUFLLEVBQUMsR0FBRyxNQUFNLENBQUM7SUFDM0IsTUFBTSxFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUUxRCxNQUFNLFFBQVEsR0FBRyxZQUFZLENBQUMsaUJBQWlCLENBQzNDLEtBQUssQ0FBQyxLQUFpRCxFQUFFLFVBQVUsRUFDbkUsT0FBTyxFQUFFLGNBQWMsQ0FBQSxDQUFDLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQ3BELE1BQU0sRUFBRSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQUMsS0FBSyxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUM7SUFFeEQsaUJBQWlCLENBQ2IsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsRUFDdEMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsRUFDbkMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLEVBQUUsRUFDbkMsUUFBUSxDQUFDLFNBQVM7SUFDbEIsd0VBQXdFO0lBQ3hFLGdEQUFnRDtJQUNoRCxnQkFBZ0IsQ0FBQSxRQUFRLENBQUMsVUFBVSxFQUNuQyxRQUFRLENBQUMsT0FBTyxFQUNoQixRQUFRLENBQUMsUUFBUSxFQUNqQixRQUFRLENBQUMsT0FBTyxFQUNoQixRQUFRLENBQUMsUUFBUSxFQUNqQixRQUFRLENBQUMsU0FBUyxFQUNsQixRQUFRLENBQUMsUUFBUSxFQUNqQixRQUFRLENBQUMsV0FBVyxFQUNwQixRQUFRLENBQUMsWUFBWSxFQUNyQixRQUFRLENBQUMsV0FBVyxFQUNwQixRQUFRLENBQUMsYUFBYSxFQUN0QixRQUFRLENBQUMsY0FBYyxFQUN2QixRQUFRLENBQUMsYUFBYSxFQUN0QixRQUFRLENBQUMsb0JBQW9CLEVBQzdCLFFBQVEsQ0FBQyxxQkFBcUIsRUFDOUIsUUFBUSxDQUFDLG9CQUFvQixFQUM3QixRQUFRLENBQUMsT0FBTyxDQUFDLEtBQUssRUFDdEIsUUFBUSxDQUFDLE9BQU8sQ0FBQyxHQUFHLEVBQ3BCLFFBQVEsQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUN4QixDQUFDO0lBQ0YsT0FBTyxFQUFFLENBQUM7QUFDWixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sbUJBQW1CLEdBQWlCO0lBQy9DLFVBQVUsRUFBRSxhQUFhO0lBQ3pCLFdBQVcsRUFBRSxNQUFNO0lBQ25CLFNBQVMsRUFBRSxLQUFLO0lBQ2hCLFVBQVUsRUFBRSxhQUFzQztDQUNuRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBNYXhQb29sM0RHcmFkLCBNYXhQb29sM0RHcmFkQXR0cnMsIE1heFBvb2wzREdyYWRJbnB1dHMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmFja2VuZFdhc219IGZyb20gJy4uL2JhY2tlbmRfd2FzbSc7XG5cbmxldCB3YXNtTWF4UG9vbDNER3JhZDogKFxuICAgIHhJZDogbnVtYmVyLCBkeUlkOiBudW1iZXIsIGR4SWQ6IG51bWJlciwgYmF0Y2hTaXplOiBudW1iZXIsXG4gICAgY2hhbm5lbFNpemU6IG51bWJlciwgaW5EZXB0aDogbnVtYmVyLCBpbkhlaWdodDogbnVtYmVyLCBpbldpZHRoOiBudW1iZXIsXG4gICAgb3V0RGVwdGg6IG51bWJlciwgb3V0SGVpZ2h0OiBudW1iZXIsIG91dFdpZHRoOiBudW1iZXIsIHN0cmlkZURlcHRoOiBudW1iZXIsXG4gICAgc3RyaWRlSGVpZ2h0OiBudW1iZXIsIHN0cmlkZVdpZHRoOiBudW1iZXIsIGRpbGF0aW9uRGVwdGg6IG51bWJlcixcbiAgICBkaWxhdGlvbkhlaWdodDogbnVtYmVyLCBkaWxhdGlvbldpZHRoOiBudW1iZXIsIGVmZmVjdGl2ZUZpbHRlckRlcHRoOiBudW1iZXIsXG4gICAgZWZmZWN0aXZlRmlsdGVySGVpZ2h0OiBudW1iZXIsIGVmZmVjdGl2ZUZpbHRlcldpZHRoOiBudW1iZXIsXG4gICAgcGFkRnJvbnQ6IG51bWJlciwgcGFkVG9wOiBudW1iZXIsIHBhZExlZnQ6IG51bWJlcikgPT4gdm9pZDtcblxuZnVuY3Rpb24gc2V0dXAoYmFja2VuZDogQmFja2VuZFdhc20pIHtcbiAgd2FzbU1heFBvb2wzREdyYWQgPSBiYWNrZW5kLndhc20uY3dyYXAoJ01heFBvb2wzREdyYWQnLCBudWxsLCBbXG4gICAgJ251bWJlcicsICAvLyB4SWRcbiAgICAnbnVtYmVyJywgIC8vIGR5SWRcbiAgICAnbnVtYmVyJywgIC8vIGR4SWRcbiAgICAnbnVtYmVyJywgIC8vIGJhdGNoU2l6ZVxuICAgICdudW1iZXInLCAgLy8gY2hhbm5lbFNpemVcbiAgICAnbnVtYmVyJywgIC8vIGluRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIGluSGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBpbldpZHRoXG4gICAgJ251bWJlcicsICAvLyBvdXREZXB0aFxuICAgICdudW1iZXInLCAgLy8gb3V0SGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBvdXRXaWR0aFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIHN0cmlkZUhlaWdodFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uSGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBkaWxhdGlvbldpZHRoXG4gICAgJ251bWJlcicsICAvLyBlZmZlY3RpdmVGaWx0ZXJEZXB0aFxuICAgICdudW1iZXInLCAgLy8gZWZmZWN0aXZlRmlsdGVySGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBlZmZlY3RpdmVGaWx0ZXJXaWR0aFxuICAgICdudW1iZXInLCAgLy8gcGFkRnJvbnRcbiAgICAnbnVtYmVyJywgIC8vIHBhZFRvcFxuICAgICdudW1iZXInLCAgLy8gcGFkTGVmdFxuICBdKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2wzREdyYWQoYXJnczoge1xuICBpbnB1dHM6IE1heFBvb2wzREdyYWRJbnB1dHMsXG4gIGF0dHJzOiBNYXhQb29sM0RHcmFkQXR0cnMsXG4gIGJhY2tlbmQ6IEJhY2tlbmRXYXNtLFxufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7ZHksIGlucHV0fSA9IGlucHV0cztcbiAgY29uc3Qge2ZpbHRlclNpemUsIHN0cmlkZXMsIHBhZCwgZGltUm91bmRpbmdNb2RlfSA9IGF0dHJzO1xuXG4gIGNvbnN0IGNvbnZJbmZvID0gYmFja2VuZF91dGlsLmNvbXB1dGVQb29sM0RJbmZvKFxuICAgICAgaW5wdXQuc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmlsdGVyU2l6ZSxcbiAgICAgIHN0cmlkZXMsIC8qZGlsYXRpb25zPSovMSwgcGFkLCBkaW1Sb3VuZGluZ01vZGUpO1xuICBjb25zdCBkeCA9IGJhY2tlbmQubWFrZU91dHB1dChpbnB1dC5zaGFwZSwgaW5wdXQuZHR5cGUpO1xuXG4gIHdhc21NYXhQb29sM0RHcmFkKFxuICAgICAgYmFja2VuZC5kYXRhSWRNYXAuZ2V0KGlucHV0LmRhdGFJZCkuaWQsXG4gICAgICBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoZHkuZGF0YUlkKS5pZCxcbiAgICAgIGJhY2tlbmQuZGF0YUlkTWFwLmdldChkeC5kYXRhSWQpLmlkLFxuICAgICAgY29udkluZm8uYmF0Y2hTaXplLFxuICAgICAgLy8gU2luY2UgUG9vbDNEIG9wcyAoTWF4UG9vbDNEIGFuZCBNYXhQb29sM0QpIHN1cHBvcnQgM0QgZmlsdGVyIG9ubHksIGluXG4gICAgICAvLyBjaGFubmVscyBzaG91bGQgYWx3YXlzIGVxdWFsIHRvIG91dCBjaGFubmVscy5cbiAgICAgIC8qY2hhbm5lbFNpemU9Ki9jb252SW5mby5pbkNoYW5uZWxzLFxuICAgICAgY29udkluZm8uaW5EZXB0aCxcbiAgICAgIGNvbnZJbmZvLmluSGVpZ2h0LFxuICAgICAgY29udkluZm8uaW5XaWR0aCxcbiAgICAgIGNvbnZJbmZvLm91dERlcHRoLFxuICAgICAgY29udkluZm8ub3V0SGVpZ2h0LFxuICAgICAgY29udkluZm8ub3V0V2lkdGgsXG4gICAgICBjb252SW5mby5zdHJpZGVEZXB0aCxcbiAgICAgIGNvbnZJbmZvLnN0cmlkZUhlaWdodCxcbiAgICAgIGNvbnZJbmZvLnN0cmlkZVdpZHRoLFxuICAgICAgY29udkluZm8uZGlsYXRpb25EZXB0aCxcbiAgICAgIGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LFxuICAgICAgY29udkluZm8uZGlsYXRpb25XaWR0aCxcbiAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckRlcHRoLFxuICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0LFxuICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGgsXG4gICAgICBjb252SW5mby5wYWRJbmZvLmZyb250LFxuICAgICAgY29udkluZm8ucGFkSW5mby50b3AsXG4gICAgICBjb252SW5mby5wYWRJbmZvLmxlZnQsXG4gICk7XG4gIHJldHVybiBkeDtcbn1cblxuZXhwb3J0IGNvbnN0IG1heFBvb2wzREdyYWRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTWF4UG9vbDNER3JhZCxcbiAgYmFja2VuZE5hbWU6ICd3YXNtJyxcbiAgc2V0dXBGdW5jOiBzZXR1cCxcbiAga2VybmVsRnVuYzogbWF4UG9vbDNER3JhZCBhcyB1bmtub3duIGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=