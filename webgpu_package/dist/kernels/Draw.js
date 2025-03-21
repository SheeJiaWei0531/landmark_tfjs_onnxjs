/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
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
import { Draw, } from '@tensorflow/tfjs-core';
import { DrawProgram } from '../draw_webgpu';
export function draw(args) {
    const { inputs, backend, attrs } = args;
    const { image } = inputs;
    const { canvas, options } = attrs;
    const [height, width] = image.shape.slice(0, 2);
    const { imageOptions } = options || {};
    const alpha = (imageOptions === null || imageOptions === void 0 ? void 0 : imageOptions.alpha) || 1;
    //  'rgba8unorm' should work on macOS according to
    //  https://bugs.chromium.org/p/chromium/issues/detail?id=1298618. But
    //  failed on macOS/M2. So use 'bgra8unorm' first when available.
    const format = backend.device.features.has('bgra8unorm-storage') ?
        'bgra8unorm' :
        'rgba8unorm';
    const outShape = [height, width];
    const program = new DrawProgram(outShape, image.dtype, format);
    canvas.width = width;
    canvas.height = height;
    const backendName = 'webgpu';
    let gpuContext = canvas.getContext(backendName);
    let canvasWebGPU;
    if (!gpuContext) {
        canvasWebGPU = new OffscreenCanvas(width, height);
        gpuContext = canvasWebGPU.getContext(backendName);
    }
    const numChannels = image.shape.length === 3 ? image.shape[2] : 1;
    gpuContext.configure({
        device: backend.device,
        format,
        usage: GPUTextureUsage.STORAGE_BINDING,
        alphaMode: 'premultiplied'
    });
    const outputDtype = 'int32';
    const output = backend.makeTensorInfo(outShape, outputDtype);
    const info = backend.tensorMap.get(output.dataId);
    info.resource = gpuContext.getCurrentTexture();
    info.external = true;
    const uniformData = [{ type: 'uint32', data: [numChannels] }, { type: 'float32', data: [alpha] }];
    backend.runWebGPUProgram(program, [image], outputDtype, uniformData, output);
    if (canvasWebGPU) {
        const canvas2dContext = canvas.getContext('2d');
        if (!canvas2dContext) {
            throw new Error(`Please make sure this canvas has only been used for 2d or webgpu context!`);
        }
        canvas2dContext.drawImage(canvasWebGPU, 0, 0);
    }
    backend.disposeData(output.dataId);
    return image;
}
export const drawConfig = {
    kernelName: Draw,
    backendName: 'webgpu',
    kernelFunc: draw
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiRHJhdy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJncHUvc3JjL2tlcm5lbHMvRHJhdy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsSUFBSSxHQUF5QixNQUFNLHVCQUF1QixDQUFDO0FBR25FLE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUUzQyxNQUFNLFVBQVUsSUFBSSxDQUNoQixJQUFvRTtJQUV0RSxNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUMsR0FBRyxJQUFJLENBQUM7SUFDdEMsTUFBTSxFQUFDLEtBQUssRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUN2QixNQUFNLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBQyxHQUFHLEtBQUssQ0FBQztJQUNoQyxNQUFNLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNoRCxNQUFNLEVBQUMsWUFBWSxFQUFDLEdBQUcsT0FBTyxJQUFJLEVBQUUsQ0FBQztJQUNyQyxNQUFNLEtBQUssR0FBRyxDQUFBLFlBQVksYUFBWixZQUFZLHVCQUFaLFlBQVksQ0FBRyxLQUFLLEtBQUksQ0FBQyxDQUFDO0lBRXhDLGtEQUFrRDtJQUNsRCxzRUFBc0U7SUFDdEUsaUVBQWlFO0lBQ2pFLE1BQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsUUFBUSxDQUFDLEdBQUcsQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDLENBQUM7UUFDOUQsWUFBWSxDQUFDLENBQUM7UUFDZCxZQUFZLENBQUM7SUFDakIsTUFBTSxRQUFRLEdBQUcsQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDakMsTUFBTSxPQUFPLEdBQUcsSUFBSSxXQUFXLENBQUMsUUFBUSxFQUFFLEtBQUssQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDL0QsTUFBTSxDQUFDLEtBQUssR0FBRyxLQUFLLENBQUM7SUFDckIsTUFBTSxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7SUFDdkIsTUFBTSxXQUFXLEdBQUcsUUFBUSxDQUFDO0lBQzdCLElBQUksVUFBVSxHQUFHLE1BQU0sQ0FBQyxVQUFVLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDaEQsSUFBSSxZQUFZLENBQUM7SUFDakIsSUFBSSxDQUFDLFVBQVUsRUFBRTtRQUNmLFlBQVksR0FBRyxJQUFJLGVBQWUsQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7UUFDbEQsVUFBVSxHQUFHLFlBQVksQ0FBQyxVQUFVLENBQUMsV0FBVyxDQUFDLENBQUM7S0FDbkQ7SUFDRCxNQUFNLFdBQVcsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNsRSxVQUFVLENBQUMsU0FBUyxDQUFDO1FBQ25CLE1BQU0sRUFBRSxPQUFPLENBQUMsTUFBTTtRQUN0QixNQUFNO1FBQ04sS0FBSyxFQUFFLGVBQWUsQ0FBQyxlQUFlO1FBQ3RDLFNBQVMsRUFBRSxlQUFlO0tBQzNCLENBQUMsQ0FBQztJQUVILE1BQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQztJQUM1QixNQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsY0FBYyxDQUFDLFFBQVEsRUFBRSxXQUFXLENBQUMsQ0FBQztJQUM3RCxNQUFNLElBQUksR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDbEQsSUFBSSxDQUFDLFFBQVEsR0FBRyxVQUFVLENBQUMsaUJBQWlCLEVBQUUsQ0FBQztJQUMvQyxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQztJQUVyQixNQUFNLFdBQVcsR0FDYixDQUFDLEVBQUMsSUFBSSxFQUFFLFFBQVEsRUFBRSxJQUFJLEVBQUUsQ0FBQyxXQUFXLENBQUMsRUFBQyxFQUFFLEVBQUMsSUFBSSxFQUFFLFNBQVMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxLQUFLLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDOUUsT0FBTyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRSxDQUFDLEtBQUssQ0FBQyxFQUFFLFdBQVcsRUFBRSxXQUFXLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFFN0UsSUFBSSxZQUFZLEVBQUU7UUFDaEIsTUFBTSxlQUFlLEdBQUcsTUFBTSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNoRCxJQUFJLENBQUMsZUFBZSxFQUFFO1lBQ3BCLE1BQU0sSUFBSSxLQUFLLENBQ1gsMkVBQTJFLENBQUMsQ0FBQztTQUNsRjtRQUNELGVBQWUsQ0FBQyxTQUFTLENBQUMsWUFBWSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztLQUMvQztJQUNELE9BQU8sQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ25DLE9BQU8sS0FBSyxDQUFDO0FBQ2YsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFVBQVUsR0FBaUI7SUFDdEMsVUFBVSxFQUFFLElBQUk7SUFDaEIsV0FBVyxFQUFFLFFBQVE7SUFDckIsVUFBVSxFQUFFLElBQTZCO0NBQzFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMyBHb29nbGUgTExDLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSBiYWNrZW5kIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtEcmF3LCBEcmF3QXR0cnMsIERyYXdJbnB1dHMsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge1dlYkdQVUJhY2tlbmR9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ3B1JztcbmltcG9ydCB7RHJhd1Byb2dyYW19IGZyb20gJy4uL2RyYXdfd2ViZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGRyYXcoXG4gICAgYXJnczoge2lucHV0czogRHJhd0lucHV0cywgYmFja2VuZDogV2ViR1BVQmFja2VuZCwgYXR0cnM6IERyYXdBdHRyc30pOlxuICAgIFRlbnNvckluZm8ge1xuICBjb25zdCB7aW5wdXRzLCBiYWNrZW5kLCBhdHRyc30gPSBhcmdzO1xuICBjb25zdCB7aW1hZ2V9ID0gaW5wdXRzO1xuICBjb25zdCB7Y2FudmFzLCBvcHRpb25zfSA9IGF0dHJzO1xuICBjb25zdCBbaGVpZ2h0LCB3aWR0aF0gPSBpbWFnZS5zaGFwZS5zbGljZSgwLCAyKTtcbiAgY29uc3Qge2ltYWdlT3B0aW9uc30gPSBvcHRpb25zIHx8IHt9O1xuICBjb25zdCBhbHBoYSA9IGltYWdlT3B0aW9ucyA/LmFscGhhIHx8IDE7XG5cbiAgLy8gICdyZ2JhOHVub3JtJyBzaG91bGQgd29yayBvbiBtYWNPUyBhY2NvcmRpbmcgdG9cbiAgLy8gIGh0dHBzOi8vYnVncy5jaHJvbWl1bS5vcmcvcC9jaHJvbWl1bS9pc3N1ZXMvZGV0YWlsP2lkPTEyOTg2MTguIEJ1dFxuICAvLyAgZmFpbGVkIG9uIG1hY09TL00yLiBTbyB1c2UgJ2JncmE4dW5vcm0nIGZpcnN0IHdoZW4gYXZhaWxhYmxlLlxuICBjb25zdCBmb3JtYXQgPSBiYWNrZW5kLmRldmljZS5mZWF0dXJlcy5oYXMoJ2JncmE4dW5vcm0tc3RvcmFnZScpID9cbiAgICAgICdiZ3JhOHVub3JtJyA6XG4gICAgICAncmdiYTh1bm9ybSc7XG4gIGNvbnN0IG91dFNoYXBlID0gW2hlaWdodCwgd2lkdGhdO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IERyYXdQcm9ncmFtKG91dFNoYXBlLCBpbWFnZS5kdHlwZSwgZm9ybWF0KTtcbiAgY2FudmFzLndpZHRoID0gd2lkdGg7XG4gIGNhbnZhcy5oZWlnaHQgPSBoZWlnaHQ7XG4gIGNvbnN0IGJhY2tlbmROYW1lID0gJ3dlYmdwdSc7XG4gIGxldCBncHVDb250ZXh0ID0gY2FudmFzLmdldENvbnRleHQoYmFja2VuZE5hbWUpO1xuICBsZXQgY2FudmFzV2ViR1BVO1xuICBpZiAoIWdwdUNvbnRleHQpIHtcbiAgICBjYW52YXNXZWJHUFUgPSBuZXcgT2Zmc2NyZWVuQ2FudmFzKHdpZHRoLCBoZWlnaHQpO1xuICAgIGdwdUNvbnRleHQgPSBjYW52YXNXZWJHUFUuZ2V0Q29udGV4dChiYWNrZW5kTmFtZSk7XG4gIH1cbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSBpbWFnZS5zaGFwZS5sZW5ndGggPT09IDMgPyBpbWFnZS5zaGFwZVsyXSA6IDE7XG4gIGdwdUNvbnRleHQuY29uZmlndXJlKHtcbiAgICBkZXZpY2U6IGJhY2tlbmQuZGV2aWNlLFxuICAgIGZvcm1hdCxcbiAgICB1c2FnZTogR1BVVGV4dHVyZVVzYWdlLlNUT1JBR0VfQklORElORyxcbiAgICBhbHBoYU1vZGU6ICdwcmVtdWx0aXBsaWVkJ1xuICB9KTtcblxuICBjb25zdCBvdXRwdXREdHlwZSA9ICdpbnQzMic7XG4gIGNvbnN0IG91dHB1dCA9IGJhY2tlbmQubWFrZVRlbnNvckluZm8ob3V0U2hhcGUsIG91dHB1dER0eXBlKTtcbiAgY29uc3QgaW5mbyA9IGJhY2tlbmQudGVuc29yTWFwLmdldChvdXRwdXQuZGF0YUlkKTtcbiAgaW5mby5yZXNvdXJjZSA9IGdwdUNvbnRleHQuZ2V0Q3VycmVudFRleHR1cmUoKTtcbiAgaW5mby5leHRlcm5hbCA9IHRydWU7XG5cbiAgY29uc3QgdW5pZm9ybURhdGEgPVxuICAgICAgW3t0eXBlOiAndWludDMyJywgZGF0YTogW251bUNoYW5uZWxzXX0sIHt0eXBlOiAnZmxvYXQzMicsIGRhdGE6IFthbHBoYV19XTtcbiAgYmFja2VuZC5ydW5XZWJHUFVQcm9ncmFtKHByb2dyYW0sIFtpbWFnZV0sIG91dHB1dER0eXBlLCB1bmlmb3JtRGF0YSwgb3V0cHV0KTtcblxuICBpZiAoY2FudmFzV2ViR1BVKSB7XG4gICAgY29uc3QgY2FudmFzMmRDb250ZXh0ID0gY2FudmFzLmdldENvbnRleHQoJzJkJyk7XG4gICAgaWYgKCFjYW52YXMyZENvbnRleHQpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgUGxlYXNlIG1ha2Ugc3VyZSB0aGlzIGNhbnZhcyBoYXMgb25seSBiZWVuIHVzZWQgZm9yIDJkIG9yIHdlYmdwdSBjb250ZXh0IWApO1xuICAgIH1cbiAgICBjYW52YXMyZENvbnRleHQuZHJhd0ltYWdlKGNhbnZhc1dlYkdQVSwgMCwgMCk7XG4gIH1cbiAgYmFja2VuZC5kaXNwb3NlRGF0YShvdXRwdXQuZGF0YUlkKTtcbiAgcmV0dXJuIGltYWdlO1xufVxuXG5leHBvcnQgY29uc3QgZHJhd0NvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBEcmF3LFxuICBiYWNrZW5kTmFtZTogJ3dlYmdwdScsXG4gIGtlcm5lbEZ1bmM6IGRyYXcgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jXG59O1xuIl19