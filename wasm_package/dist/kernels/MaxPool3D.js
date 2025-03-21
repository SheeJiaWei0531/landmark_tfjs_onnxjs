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
import { backend_util, MaxPool3D } from '@tensorflow/tfjs-core';
let wasmMaxPool3D;
function setup(backend) {
    wasmMaxPool3D = backend.wasm.cwrap('MaxPool3D', null, [
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
export function maxPool3D(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode, dataFormat } = attrs;
    const convInfo = backend_util.computePool3DInfo(x.shape, filterSize, strides, 
    /*dilations=*/ 1, pad, dimRoundingMode, dataFormat);
    const out = backend.makeOutput(convInfo.outShape, x.dtype);
    wasmMaxPool3D(backend.dataIdMap.get(x.dataId).id, backend.dataIdMap.get(out.dataId).id, convInfo.batchSize, 
    // Since Pool3D ops (AvgPool3D and MaxPool3D) support 3D filter only, in
    // channels should always equal to out channels.
    /*channelSize=*/ convInfo.inChannels, convInfo.inDepth, convInfo.inHeight, convInfo.inWidth, convInfo.outDepth, convInfo.outHeight, convInfo.outWidth, convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth, convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth, convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left);
    return out;
}
export const maxPool3DConfig = {
    kernelName: MaxPool3D,
    backendName: 'wasm',
    setupFunc: setup,
    kernelFunc: maxPool3D
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWF4UG9vbDNELmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbHMvTWF4UG9vbDNELnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxZQUFZLEVBQTRCLFNBQVMsRUFBOEMsTUFBTSx1QkFBdUIsQ0FBQztBQUlySSxJQUFJLGFBTzBELENBQUM7QUFFL0QsU0FBUyxLQUFLLENBQUMsT0FBb0I7SUFDakMsYUFBYSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLFdBQVcsRUFBRSxJQUFJLEVBQUU7UUFDcEQsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUTtRQUNSLFFBQVE7UUFDUixRQUFRO1FBQ1IsUUFBUSxFQUFHLFVBQVU7S0FDdEIsQ0FBQyxDQUFDO0FBQ0wsQ0FBQztBQUVELE1BQU0sVUFBVSxTQUFTLENBQUMsSUFJekI7SUFDQyxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLENBQUMsRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUNuQixNQUFNLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFFLFVBQVUsRUFBQyxHQUFHLEtBQUssQ0FBQztJQUV0RSxNQUFNLFFBQVEsR0FBRyxZQUFZLENBQUMsaUJBQWlCLENBQzNDLENBQUMsQ0FBQyxLQUFpRCxFQUFFLFVBQVUsRUFBRSxPQUFPO0lBQ3hFLGNBQWMsQ0FBQSxDQUFDLEVBQUUsR0FBRyxFQUFFLGVBQWUsRUFBRSxVQUFVLENBQUMsQ0FBQztJQUN2RCxNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFDLFFBQVEsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRTNELGFBQWEsQ0FDVCxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxFQUNsQyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxFQUNwQyxRQUFRLENBQUMsU0FBUztJQUNsQix3RUFBd0U7SUFDeEUsZ0RBQWdEO0lBQ2hELGdCQUFnQixDQUFBLFFBQVEsQ0FBQyxVQUFVLEVBQ25DLFFBQVEsQ0FBQyxPQUFPLEVBQ2hCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxPQUFPLEVBQ2hCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxTQUFTLEVBQ2xCLFFBQVEsQ0FBQyxRQUFRLEVBQ2pCLFFBQVEsQ0FBQyxXQUFXLEVBQ3BCLFFBQVEsQ0FBQyxZQUFZLEVBQ3JCLFFBQVEsQ0FBQyxXQUFXLEVBQ3BCLFFBQVEsQ0FBQyxhQUFhLEVBQ3RCLFFBQVEsQ0FBQyxjQUFjLEVBQ3ZCLFFBQVEsQ0FBQyxhQUFhLEVBQ3RCLFFBQVEsQ0FBQyxvQkFBb0IsRUFDN0IsUUFBUSxDQUFDLHFCQUFxQixFQUM5QixRQUFRLENBQUMsb0JBQW9CLEVBQzdCLFFBQVEsQ0FBQyxPQUFPLENBQUMsS0FBSyxFQUN0QixRQUFRLENBQUMsT0FBTyxDQUFDLEdBQUcsRUFDcEIsUUFBUSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQ3hCLENBQUM7SUFDRixPQUFPLEdBQUcsQ0FBQztBQUNiLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxlQUFlLEdBQWlCO0lBQzNDLFVBQVUsRUFBRSxTQUFTO0lBQ3JCLFdBQVcsRUFBRSxNQUFNO0lBQ25CLFNBQVMsRUFBRSxLQUFLO0lBQ2hCLFVBQVUsRUFBRSxTQUFrQztDQUMvQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjMgR29vZ2xlIExMQy5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgS2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBNYXhQb29sM0QsIE1heFBvb2wzREF0dHJzLCBNYXhQb29sM0RJbnB1dHMsIFRlbnNvckluZm99IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7QmFja2VuZFdhc219IGZyb20gJy4uL2JhY2tlbmRfd2FzbSc7XG5cbmxldCB3YXNtTWF4UG9vbDNEOiAoXG4gICAgeElkOiBudW1iZXIsIG91dElkOiBudW1iZXIsIGJhdGNoU2l6ZTogbnVtYmVyLCBjaGFubmVsU2l6ZTogbnVtYmVyLFxuICAgIGluRGVwdGg6IG51bWJlciwgaW5IZWlnaHQ6IG51bWJlciwgaW5XaWR0aDogbnVtYmVyLCBvdXREZXB0aDogbnVtYmVyLFxuICAgIG91dEhlaWdodDogbnVtYmVyLCBvdXRXaWR0aDogbnVtYmVyLCBzdHJpZGVEZXB0aDogbnVtYmVyLFxuICAgIHN0cmlkZUhlaWdodDogbnVtYmVyLCBzdHJpZGVXaWR0aDogbnVtYmVyLCBkaWxhdGlvbkRlcHRoOiBudW1iZXIsXG4gICAgZGlsYXRpb25IZWlnaHQ6IG51bWJlciwgZGlsYXRpb25XaWR0aDogbnVtYmVyLCBlZmZlY3RpdmVGaWx0ZXJEZXB0aDogbnVtYmVyLFxuICAgIGVmZmVjdGl2ZUZpbHRlckhlaWdodDogbnVtYmVyLCBlZmZlY3RpdmVGaWx0ZXJXaWR0aDogbnVtYmVyLFxuICAgIHBhZEZyb250OiBudW1iZXIsIHBhZFRvcDogbnVtYmVyLCBwYWRMZWZ0OiBudW1iZXIpID0+IHZvaWQ7XG5cbmZ1bmN0aW9uIHNldHVwKGJhY2tlbmQ6IEJhY2tlbmRXYXNtKSB7XG4gIHdhc21NYXhQb29sM0QgPSBiYWNrZW5kLndhc20uY3dyYXAoJ01heFBvb2wzRCcsIG51bGwsIFtcbiAgICAnbnVtYmVyJywgIC8vIHhJZFxuICAgICdudW1iZXInLCAgLy8gb3V0SWRcbiAgICAnbnVtYmVyJywgIC8vIGJhdGNoU2l6ZVxuICAgICdudW1iZXInLCAgLy8gY2hhbm5lbFNpemVcbiAgICAnbnVtYmVyJywgIC8vIGluRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIGluSGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBpbldpZHRoXG4gICAgJ251bWJlcicsICAvLyBvdXREZXB0aFxuICAgICdudW1iZXInLCAgLy8gb3V0SGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBvdXRXaWR0aFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIHN0cmlkZUhlaWdodFxuICAgICdudW1iZXInLCAgLy8gc3RyaWRlV2lkdGhcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uRGVwdGhcbiAgICAnbnVtYmVyJywgIC8vIGRpbGF0aW9uSGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBkaWxhdGlvbldpZHRoXG4gICAgJ251bWJlcicsICAvLyBlZmZlY3RpdmVGaWx0ZXJEZXB0aFxuICAgICdudW1iZXInLCAgLy8gZWZmZWN0aXZlRmlsdGVySGVpZ2h0XG4gICAgJ251bWJlcicsICAvLyBlZmZlY3RpdmVGaWx0ZXJXaWR0aFxuICAgICdudW1iZXInLCAgLy8gcGFkRnJvbnRcbiAgICAnbnVtYmVyJywgIC8vIHBhZFRvcFxuICAgICdudW1iZXInLCAgLy8gcGFkTGVmdFxuICBdKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2wzRChhcmdzOiB7XG4gIGlucHV0czogTWF4UG9vbDNESW5wdXRzLFxuICBhdHRyczogTWF4UG9vbDNEQXR0cnMsXG4gIGJhY2tlbmQ6IEJhY2tlbmRXYXNtLFxufSk6IFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7eH0gPSBpbnB1dHM7XG4gIGNvbnN0IHtmaWx0ZXJTaXplLCBzdHJpZGVzLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSwgZGF0YUZvcm1hdH0gPSBhdHRycztcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlUG9vbDNESW5mbyhcbiAgICAgIHguc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmlsdGVyU2l6ZSwgc3RyaWRlcyxcbiAgICAgIC8qZGlsYXRpb25zPSovMSwgcGFkLCBkaW1Sb3VuZGluZ01vZGUsIGRhdGFGb3JtYXQpO1xuICBjb25zdCBvdXQgPSBiYWNrZW5kLm1ha2VPdXRwdXQoY29udkluZm8ub3V0U2hhcGUsIHguZHR5cGUpO1xuXG4gIHdhc21NYXhQb29sM0QoXG4gICAgICBiYWNrZW5kLmRhdGFJZE1hcC5nZXQoeC5kYXRhSWQpLmlkLFxuICAgICAgYmFja2VuZC5kYXRhSWRNYXAuZ2V0KG91dC5kYXRhSWQpLmlkLFxuICAgICAgY29udkluZm8uYmF0Y2hTaXplLFxuICAgICAgLy8gU2luY2UgUG9vbDNEIG9wcyAoQXZnUG9vbDNEIGFuZCBNYXhQb29sM0QpIHN1cHBvcnQgM0QgZmlsdGVyIG9ubHksIGluXG4gICAgICAvLyBjaGFubmVscyBzaG91bGQgYWx3YXlzIGVxdWFsIHRvIG91dCBjaGFubmVscy5cbiAgICAgIC8qY2hhbm5lbFNpemU9Ki9jb252SW5mby5pbkNoYW5uZWxzLFxuICAgICAgY29udkluZm8uaW5EZXB0aCxcbiAgICAgIGNvbnZJbmZvLmluSGVpZ2h0LFxuICAgICAgY29udkluZm8uaW5XaWR0aCxcbiAgICAgIGNvbnZJbmZvLm91dERlcHRoLFxuICAgICAgY29udkluZm8ub3V0SGVpZ2h0LFxuICAgICAgY29udkluZm8ub3V0V2lkdGgsXG4gICAgICBjb252SW5mby5zdHJpZGVEZXB0aCxcbiAgICAgIGNvbnZJbmZvLnN0cmlkZUhlaWdodCxcbiAgICAgIGNvbnZJbmZvLnN0cmlkZVdpZHRoLFxuICAgICAgY29udkluZm8uZGlsYXRpb25EZXB0aCxcbiAgICAgIGNvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LFxuICAgICAgY29udkluZm8uZGlsYXRpb25XaWR0aCxcbiAgICAgIGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckRlcHRoLFxuICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVySGVpZ2h0LFxuICAgICAgY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGgsXG4gICAgICBjb252SW5mby5wYWRJbmZvLmZyb250LFxuICAgICAgY29udkluZm8ucGFkSW5mby50b3AsXG4gICAgICBjb252SW5mby5wYWRJbmZvLmxlZnQsXG4gICk7XG4gIHJldHVybiBvdXQ7XG59XG5cbmV4cG9ydCBjb25zdCBtYXhQb29sM0RDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTWF4UG9vbDNELFxuICBiYWNrZW5kTmFtZTogJ3dhc20nLFxuICBzZXR1cEZ1bmM6IHNldHVwLFxuICBrZXJuZWxGdW5jOiBtYXhQb29sM0QgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19