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
import { AvgPool3DGrad, backend_util } from '@tensorflow/tfjs-core';
let wasmAvgPool3DGrad;
function setup(backend) {
    wasmAvgPool3DGrad = backend.wasm.cwrap('AvgPool3DGrad', null, [
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
        'number',
        'number',
        'number', // filterWidth
    ]);
}
export function avgPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = backend_util.computePool3DInfo(input.shape, filterSize, strides, /*dilations=*/ 1, pad, dimRoundingMode);
    const dx = backend.makeOutput(input.shape, input.dtype);
    wasmAvgPool3DGrad(backend.dataIdMap.get(dy.dataId).id, backend.dataIdMap.get(dx.dataId).id, convInfo.batchSize, 
    // Since Pool3D ops (AvgPool3D and MaxPool3D) support 3D filter only, in
    // channels should always equal to out channels.
    /*channelSize=*/ convInfo.inChannels, convInfo.inDepth, convInfo.inHeight, convInfo.inWidth, convInfo.outDepth, convInfo.outHeight, convInfo.outWidth, convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth, convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth, convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left, convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth);
    return dx;
}
export const avgPool3DGradConfig = {
    kernelName: AvgPool3DGrad,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: avgPool3DGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQXZnUG9vbDNER3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13YXNtL3NyYy9rZXJuZWxzL0F2Z1Bvb2wzREdyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLGFBQWEsRUFBMkMsWUFBWSxFQUF1QyxNQUFNLHVCQUF1QixDQUFDO0FBSWpKLElBQUksaUJBUWtELENBQUM7QUFFdkQsU0FBUyxLQUFLLENBQUMsT0FBb0I7SUFDakMsaUJBQWlCLEdBQUcsT0FBTyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsZUFBZSxFQUFFLElBQUksRUFBRTtRQUM1RCxRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRLEVBQUcsY0FBYztLQUMxQixDQUFDLENBQUM7QUFDTCxDQUFDO0FBRUQsTUFBTSxVQUFVLGFBQWEsQ0FBQyxJQUk3QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsRUFBRSxFQUFFLEtBQUssRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUMzQixNQUFNLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRTFELE1BQU0sUUFBUSxHQUFHLFlBQVksQ0FBQyxpQkFBaUIsQ0FDM0MsS0FBSyxDQUFDLEtBQWlELEVBQUUsVUFBVSxFQUNuRSxPQUFPLEVBQUUsY0FBYyxDQUFBLENBQUMsRUFBRSxHQUFHLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDcEQsTUFBTSxFQUFFLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUV4RCxpQkFBaUIsQ0FDYixPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxFQUNuQyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxFQUNuQyxRQUFRLENBQUMsU0FBUztJQUNsQix3RUFBd0U7SUFDeEUsZ0RBQWdEO0lBQ2hELGdCQUFnQixDQUFBLFFBQVEsQ0FBQyxVQUFVLEVBQ25DLFFBQVEsQ0FBQyxPQUFPLEVBQ2hCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxPQUFPLEVBQ2hCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxTQUFTLEVBQ2xCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxXQUFXLEVBQ3BCLFFBQVEsQ0FBQyxZQUFZLEVBQ3JCLFFBQVEsQ0FBQyxXQUFXLEVBQ3BCLFFBQVEsQ0FBQyxhQUFhLEVBQ3RCLFFBQVEsQ0FBQyxjQUFjLEVBQ3ZCLFFBQVEsQ0FBQyxhQUFhLEVBQ3RCLFFBQVEsQ0FBQyxvQkFBb0IsRUFDN0IsUUFBUSxDQUFDLHFCQUFxQixFQUM5QixRQUFRLENBQUMsb0JBQW9CLEVBQzdCLFFBQVEsQ0FBQyxPQUFPLENBQUMsS0FBSyxFQUN0QixRQUFRLENBQUMsT0FBTyxDQUFDLEdBQUcsRUFDcEIsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLEVBQ3JCLFFBQVEsQ0FBQyxXQUFXLEVBQ3BCLFFBQVEsQ0FBQyxZQUFZLEVBQ3JCLFFBQVEsQ0FBQyxXQUFXLENBQ3ZCLENBQUM7SUFDRixPQUFPLEVBQUUsQ0FBQztBQUNaLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxtQkFBbUIsR0FBaUI7SUFDL0MsVUFBVSxFQUFFLGFBQWE7SUFDekIsV0FBVyxFQUFFLE1BQU07SUFDbkIsU0FBUyxFQUFFLEtBQUs7SUFDaEIsVUFBVSxFQUFFLGFBQXNDO0NBQ25ELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7QXZnUG9vbDNER3JhZCwgQXZnUG9vbDNER3JhZEF0dHJzLCBBdmdQb29sM0RHcmFkSW5wdXRzLCBiYWNrZW5kX3V0aWwsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgVGVuc29ySW5mb30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtCYWNrZW5kV2FzbX0gZnJvbSAnLi4vYmFja2VuZF93YXNtJztcblxubGV0IHdhc21BdmdQb29sM0RHcmFkOiAoXG4gICAgZHlJZDogbnVtYmVyLCBkeElkOiBudW1iZXIsIGJhdGNoU2l6ZTogbnVtYmVyLCBjaGFubmVsU2l6ZTogbnVtYmVyLFxuICAgIGluRGVwdGg6IG51bWJlciwgaW5IZWlnaHQ6IG51bWJlciwgaW5XaWR0aDogbnVtYmVyLCBvdXREZXB0aDogbnVtYmVyLFxuICAgIG91dEhlaWdodDogbnVtYmVyLCBvdXRXaWR0aDogbnVtYmVyLCBzdHJpZGVEZXB0aDogbnVtYmVyLFxuICAgIHN0cmlkZUhlaWdodDogbnVtYmVyLCBzdHJpZGVXaWR0aDogbnVtYmVyLCBkaWxhdGlvbkRlcHRoOiBudW1iZXIsXG4gICAgZGlsYXRpb25IZWlnaHQ6IG51bWJlciwgZGlsYXRpb25XaWR0aDogbnVtYmVyLCBlZmZlY3RpdmVGaWx0ZXJEZXB0aDogbnVtYmVyLFxuICAgIGVmZmVjdGl2ZUZpbHRlckhlaWdodDogbnVtYmVyLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aDogbnVtYmVyLFxuICAgIHBhZEZyb250OiBudW1iZXIsIHBhZFRvcDogbnVtYmVyLCBwYWRMZWZ0OiBudW1iZXIsIGZpbHRlckRlcHRoOiBudW1iZXIsXG4gICAgZmlsdGVySGVpZ2h0OiBudW1iZXIsIGZpbHRlcldpZHRoOiBudW1iZXIpID0+IHZvaWQ7XG5cbmZ1bmN0aW9uIHNldHVwKGJhY2tlbmQ6IEJhY2tlbmRXYXNtKSB7XG4gIHdhc21BdmdQb29sM0RHcmFkID0gYmFja2VuZC53YXNtLmN3cmFwKCdBdmdQb29sM0RHcmFkJywgbnVsbCwgW1xuICAgICdudW1iZXInLCAgLy8gZHlJZFxuICAgICdudW1iZXInLCAgLy8gZHhJZFxuICAgICdudW1iZXInLCAgLy8gYmF0Y2hTaXplXG4gICAgJ251bWJlcicsICAvLyBjaGFubmVsU2l6ZVxuICAgICdudW1iZXInLCAgLy8gaW5EZXB0aFxuICAgICdudW1iZXInLCAgLy8gaW5IZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGluV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIG91dERlcHRoXG4gICAgJ251bWJlcicsICAvLyBvdXRIZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIG91dFdpZHRoXG4gICAgJ251bWJlcicsICAvLyBzdHJpZGVEZXB0aFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlSGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBzdHJpZGVXaWR0aFxuICAgICdudW1iZXInLCAgLy8gZGlsYXRpb25EZXB0aFxuICAgICdudW1iZXInLCAgLy8gZGlsYXRpb25IZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIGVmZmVjdGl2ZUZpbHRlckRlcHRoXG4gICAgJ251bWJlcicsICAvLyBlZmZlY3RpdmVGaWx0ZXJIZWlnaHRcbiAgICAnbnVtYmVyJywgIC8vIGVmZmVjdGl2ZUZpbHRlcldpZHRoXG4gICAgJ251bWJlcicsICAvLyBwYWRGcm9udFxuICAgICdudW1iZXInLCAgLy8gcGFkVG9wXG4gICAgJ251bWJlcicsICAvLyBwYWRMZWZ0XG4gICAgJ251bWJlcicsICAvLyBmaWx0ZXJEZXB0aFxuICAgICdudW1iZXInLCAgLy8gZmlsdGVySGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBmaWx0ZXJXaWR0aFxuICBdKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGF2Z1Bvb2wzREdyYWQoYXJnczoge1xuICBpbnB1dHM6IEF2Z1Bvb2wzREdyYWRJbnB1dHMsXG4gIGF0dHJzOiBBdmdQb29sM0RHcmFkQXR0cnMsXG4gIGJhY2tlbmQ6IEJhY2tlbmRXYXNtLFxufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7ZHksIGlucHV0fSA9IGlucHV0cztcbiAgY29uc3Qge2ZpbHRlclNpemUsIHN0cmlkZXMsIHBhZCwgZGltUm91bmRpbmdNb2RlfSA9IGF0dHJzO1xuXG4gIGNvbnN0IGNvbnZJbmZvID0gYmFja2VuZF91dGlsLmNvbXB1dGVQb29sM0RJbmZvKFxuICAgICAgaW5wdXQuc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmlsdGVyU2l6ZSxcbiAgICAgIHN0cmlkZXMsIC8qZGlsYXRpb25zPSovMSwgcGFkLCBkaW1Sb3VuZGluZ01vZGUpO1xuICBjb25zdCBkeCA9IGJhY2tlbmQubWFrZU91dHB1dChpbnB1dC5zaGFwZSwgaW5wdXQuZHR5cGUpO1xuXG4gIHdhc21BdmdQb29sM0RHcmFkKFxuICAgICAgYmFja2VuZC5kYXRhSWRNYXAuZ2V0KGR5LmRhdGFJZCkuaWQsXG4gICAgICBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoZHguZGF0YUlkKS5pZCxcbiAgICAgIGNvbnZJbmZvLmJhdGNoU2l6ZSxcbiAgICAgIC8vIFNpbmNlIFBvb2wzRCBvcHMgKEF2Z1Bvb2wzRCBhbmQgTWF4UG9vbDNEKSBzdXBwb3J0IDNEIGZpbHRlciBvbmx5LCBpblxuICAgICAgLy8gY2hhbm5lbHMgc2hvdWxkIGFsd2F5cyBlcXVhbCB0byBvdXQgY2hhbm5lbHMuXG4gICAgICAvKmNoYW5uZWxTaXplPSovY29udkluZm8uaW5DaGFubmVscyxcbiAgICAgIGNvbnZJbmZvLmluRGVwdGgsXG4gICAgICBjb252SW5mby5pbkhlaWdodCxcbiAgICAgIGNvbnZJbmZvLmluV2lkdGgsXG4gICAgICBjb252SW5mby5vdXREZXB0aCxcbiAgICAgIGNvbnZJbmZvLm91dEhlaWdodCxcbiAgICAgIGNvbnZJbmZvLm91dFdpZHRoLFxuICAgICAgY29udkluZm8uc3RyaWRlRGVwdGgsXG4gICAgICBjb252SW5mby5zdHJpZGVIZWlnaHQsXG4gICAgICBjb252SW5mby5zdHJpZGVXaWR0aCxcbiAgICAgIGNvbnZJbmZvLmRpbGF0aW9uRGVwdGgsXG4gICAgICBjb252SW5mby5kaWxhdGlvbkhlaWdodCxcbiAgICAgIGNvbnZJbmZvLmRpbGF0aW9uV2lkdGgsXG4gICAgICBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJEZXB0aCxcbiAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckhlaWdodCxcbiAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlcldpZHRoLFxuICAgICAgY29udkluZm8ucGFkSW5mby5mcm9udCxcbiAgICAgIGNvbnZJbmZvLnBhZEluZm8udG9wLFxuICAgICAgY29udkluZm8ucGFkSW5mby5sZWZ0LFxuICAgICAgY29udkluZm8uZmlsdGVyRGVwdGgsXG4gICAgICBjb252SW5mby5maWx0ZXJIZWlnaHQsXG4gICAgICBjb252SW5mby5maWx0ZXJXaWR0aCxcbiAgKTtcbiAgcmV0dXJuIGR4O1xufVxuXG5leHBvcnQgY29uc3QgYXZnUG9vbDNER3JhZENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBBdmdQb29sM0RHcmFkLFxuICBiYWNrZW5kTmFtZTogJ3dhc20nLFxuICBzZXR1cEZ1bmM6IHNldHVwLFxuICBrZXJuZWxGdW5jOiBhdmdQb29sM0RHcmFkIGFzIHVua25vd24gYXMgS2VybmVsRnVuY1xufTtcbiJdfQ==