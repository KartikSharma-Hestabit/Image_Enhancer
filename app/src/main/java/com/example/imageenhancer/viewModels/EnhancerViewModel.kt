package com.example.imageenhancer.viewModels

import android.content.ComponentCallbacks2
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.content.OnTrimMemoryProvider
import androidx.lifecycle.ViewModel
import com.example.imageenhancer.ml.Esrgan
import com.example.imageenhancer.resources.Resource
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.pytorch.Device
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import org.tensorflow.lite.support.image.TensorImage
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.Calendar
import javax.inject.Inject
import kotlin.math.pow
import kotlin.math.roundToInt


data class ImageSize(val width: Int, val height: Int)


@HiltViewModel
class EnhancerViewModel @Inject constructor(@ApplicationContext val context: Context) :
    ViewModel() {

    var paddingWidth: Int = 0
    var paddingHeight: Int = 0
    private val _moduleFlow = MutableStateFlow<Resource<Bitmap>?>(null)
    val moduleFlow: StateFlow<Resource<Bitmap>?> = _moduleFlow
    val maxRowColum = 6
    lateinit var model: Esrgan

    private var row = 0
    private var colum = 0
    private var paddingAdded = Pair(0, 0)
    suspend fun waternet(bitmap: Bitmap, onTrimMemory: () -> Unit) = withContext(Dispatchers.IO) {

        _moduleFlow.value = Resource.Loading("Working on Color Correction...")

        Log.d(
            "moduleStats",
            "Waternet started: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                Calendar.getInstance().get(Calendar.SECOND)
            }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
        )

        var model: Module? = LiteModuleLoader.load(assetFilePath(context, "waternet.ptl"))
        var shape = longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong())

        val wbBitmap = whiteBalanceTransform(bitmap)
        val gammaBitmap = gammaCorrection(bitmap, 0.7f)
        val heBitmap = histogramEqualization(bitmap)

        var rgbTensor = Tensor.fromBlob(bitmapToRgbNorm(bitmap), shape)
        var wbTensor = Tensor.fromBlob(bitmapToRgbNorm(wbBitmap), shape)
        var gcTensor = Tensor.fromBlob(bitmapToRgbNorm(gammaBitmap), shape)
        var heTensor = Tensor.fromBlob(bitmapToRgbNorm(heBitmap), shape)
        val output = model!!
            .forward(
                IValue.from(rgbTensor),
                IValue.from(wbTensor),
                IValue.from(gcTensor),
                IValue.from(heTensor)
            )
            .toTensor()

        wbBitmap.recycle()
        gammaBitmap.recycle()
        heBitmap.recycle()

        model.destroy()
        model = null
        rgbTensor = null
        wbTensor = null
        gcTensor = null
        heTensor = null

        wbBitmap.recycle()
        gammaBitmap.recycle()
        heBitmap.recycle()

        System.runFinalization()
        Runtime.getRuntime().gc()
        System.gc()
        onTrimMemory()

//        swinIR(context, output)

//        delay(5000)

        scuNetLarge(context, output)

    }

    @RequiresApi(Build.VERSION_CODES.O)
    suspend fun swinIR(context: Context, inputTensor: Tensor) = withContext(Dispatchers.IO) {

        _moduleFlow.value = Resource.Loading("Working on Denoising & UpScaling...")
        val modelPath = "SwinIR_small.ptl"
        val module =
            LiteModuleLoader.load(assetFilePath(context, modelPath), null, Device.VULKAN)
        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val outputBitmap = postProcessing(outputTensor)
        module.destroy()

        _moduleFlow.value = Resource.Success(outputBitmap)
        saveMediaToStorage(outputBitmap, context)

        Log.d(
            "moduleStats",
            "SwinIR Completed: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                Calendar.getInstance().get(Calendar.SECOND)
            }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
        )

    }

    private suspend fun scuNet(context: Context, inputBitmap: Bitmap) =
        withContext(Dispatchers.IO) {

            /*_moduleFlow.value = Resource.Loading("Working on Denoising...")
            val model = LiteModuleLoader.load(assetFilePath(context, "scunet_small.ptl"))
            val inputData = preProcessing(inputBitmap)
            val outputData = arrayListOf<Int>()
            val bitmap =
                Bitmap.createBitmap(inputBitmap.width, inputBitmap.height, Bitmap.Config.ARGB_8888)
            // Process the image in chunks
            for (buffer in inputData.indices step 1920000) {
                val inputTensor = Tensor.fromBlob(
                    inputData.copyOfRange(buffer, buffer + 1920000),
                    longArrayOf(1, 3, inputBitmap.height.toLong(), inputBitmap.width.toLong())
                )
                val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
                val outputArr = tensorToBitmap(outputTensor)
                for (i in outputArr)
                    outputData.add(i)

                bitmap.setPixels(
                    outputData.toIntArray(),
                    0,
                    inputBitmap.width,
                    0,
                    0,
                    inputBitmap.width,
                    inputBitmap.height
                )

                model.destroy()
                _moduleFlow.value = Resource.Success(bitmap)
                saveMediaToStorage(bitmap, context)

                Log.d(
                    "moduleStats",
                    "scuNet Completed: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                        Calendar.getInstance().get(Calendar.SECOND)
                    }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
                )
    */
        }

    /*    private suspend fun scuNetOriginal(context: Context, inputTensor: Tensor) =
            withContext(Dispatchers.IO) {


                val module =
                    LiteModuleLoader.load(assetFilePath(context, "scunet_small.ptl"))
                val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                val outputBitmap = tensorToBitmap(outputTensor)

                _moduleFlow.value = Resource.Success(outputBitmap)


                Log.d(
                    "moduleStats",
                    "ScuNet Completed: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                        Calendar.getInstance().get(Calendar.SECOND)
                    }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
                )

            }*/

    private suspend fun scuNetLarge(context: Context, inputTensor: Tensor) =
        withContext(Dispatchers.IO) {

            _moduleFlow.value = Resource.Loading("Working on Denoising")

            Log.d(
                "moduleStats",
                "SCUnet started: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                    Calendar.getInstance().get(Calendar.SECOND)
                }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
            )

            var module: Module? =
                LiteModuleLoader.load(assetFilePath(context, "scunet1024x768.ptl"))
            var outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
            val outputBitmap = tensorToBitmap(outputTensor)

//            _moduleFlow.value = Resource.Success(outputBitmap)

            val outputScuNetAndWaterNet = removePadding(outputBitmap, paddingWidth, paddingHeight)

            val inputImage = ImageSize(
                width = outputScuNetAndWaterNet.width,
                height = outputScuNetAndWaterNet.height
            )
            val availableSizes = listOf(
                ImageSize(width = 128, height = 128),
                ImageSize(width = 256, height = 256),
                ImageSize(width = 384, height = 384),
                ImageSize(width = 512, height = 512),
                ImageSize(width = 640, height = 640),
                ImageSize(width = 768, height = 768),
                ImageSize(width = 896, height = 896),
                ImageSize(width = 1024, height = 1024)
            )

            val nearestSize = findNearestSize(
                inputWidth = inputImage.width,
                inputHeight = inputImage.height,
                availableSizes = availableSizes
            )

            paddingAdded = calculatePadding(inputImage, nearestSize.second)

            val addedPaddingBitmap = addPadding(
                originalBitmap = outputScuNetAndWaterNet,
                paddingWidth = paddingAdded.first,
                paddingHeight = paddingAdded.second
            )

            row = addedPaddingBitmap.width / 128
            colum = addedPaddingBitmap.height / 128

            Log.d(
                "Model Testing",
                "Input Size: Width${inputImage.width}x${
                    inputImage.height
                }"
            )

            Log.d(
                "Model Testing",
                "Nearest Size: Width${nearestSize.second.width}x${
                    nearestSize.second.height
                }"
            )

            Log.d(
                "Model Testing",
                "Add Padding Size: Width${paddingAdded.first}x${
                    paddingAdded.second
                }"
            )

            val newBitmap = divideBitmap(addedPaddingBitmap, maxRow = row, maxColum = colum)

            outputBitmap.recycle()
            outputTensor = null

            module.destroy()
            module = null
            System.gc()

//            delay(5000)


            executeModel(newBitmap, context)

        }

    private suspend fun executeModel(downsampled: List<Bitmap>, context: Context) =
        withContext(Dispatchers.IO) {

            _moduleFlow.value = Resource.Loading("Working on UpScaling...")


            Log.d(
                "moduleStats",
                "ESRGAN started: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                    Calendar.getInstance().get(Calendar.SECOND)
                }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
            )
            model = Esrgan.newInstance(context)


            val quarterPoint = downsampled.size / 2
            var part1 = downsampled.subList(0, quarterPoint)
            var part2 = downsampled.subList(quarterPoint, downsampled.size)


            println("Image Size: ${downsampled[0].width}, ${downsampled[0].height}")

            var mergedBitmaps: List<Bitmap> = listOf()

            runBlocking {
                val deferred1 = async { processBitmaps(part1) }
                val deferred2 = async { processBitmaps(part2) }

                val mergedBitmapsPart1 = deferred1.await()
                val mergedBitmapsPart2 = deferred2.await()

                mergedBitmaps = mergedBitmapsPart1 + mergedBitmapsPart2
            }


            val mergedBitmap = mergeBitmaps(mergedBitmaps)

            val removedPaddingFromMergedBitmap =
                removePaddingRG(mergedBitmap, paddingAdded.first * 4, paddingAdded.second * 4)

            Log.d(
                "moduleStats",
                "ESRGAN stoped: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                    Calendar.getInstance().get(Calendar.SECOND)
                }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
            )

            _moduleFlow.value = Resource.Success(removedPaddingFromMergedBitmap)

            // Releases model resources if no longer used.
            model.close()

            mergedBitmaps = emptyList()
            part1 = emptyList()
            part2 = emptyList()
            System.gc()

            saveMediaToStorage(removedPaddingFromMergedBitmap, context)
        }


    private suspend fun processBitmaps(bitmaps: List<Bitmap>): List<Bitmap> {
        return bitmaps.map { bitmap ->
            getOutput(bitmap, model)
        }
    }

    private suspend fun getOutput(bitmap: Bitmap, model: Esrgan): Bitmap {
        val originalImage = TensorImage.fromBitmap(bitmap)
        val outputs = model.process(originalImage)
        val enhancedImage = outputs.enhancedImageAsTensorImage
        val enhancedImageBitmap = enhancedImage.bitmap
        return enhancedImageBitmap
    }

    private fun mergeBitmaps(bitmaps: List<Bitmap>): Bitmap {
        // Assuming all bitmaps have the same dimensions
        val bitmapCount = bitmaps.size
        require(bitmapCount == row * colum) { "Expected ${row * colum} bitmaps, but got $bitmapCount" }
        val width = bitmaps[0].width
        val height = bitmaps[0].height
        // Calculate the dimensions of the merged bitmap
        val mergedWidth = width * row
        val mergedHeight = height * colum
        val result = Bitmap.createBitmap(mergedWidth, mergedHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        for (i in 0 until colum) {
            for (j in 0 until row) {
                val index = i * row + j
                val x = j * width
                val y = i * height
                canvas.drawBitmap(bitmaps[index], x.toFloat(), y.toFloat(), null)
            }
        }
        return result
    }
}


private fun tensorToBitmap(
    outputTensor: Tensor,
    whiteBalance: FloatArray = floatArrayOf(1.0f, 1.0f, 1.0f),
    gamma: Float = 1.0f
): Bitmap {
    // Assuming the output tensor contains image data in the correct format
    val shape = outputTensor.shape() // Get the shape of the tensor
    val channels = shape[1].toInt() // Assuming channels is the second dimension
    val height = shape[2].toInt() // Assuming height is the third dimension
    val width = shape[3].toInt() // Assuming width is the fourth dimension

    Log.e("TAGH ", " tensor shape: $channels height $height width $width")

    // Get the data as float array
    val floatBuffer = outputTensor.dataAsFloatArray

    // Transpose dimensions if necessary (e.g., from (channels, height, width) to (height, width, channels))
    val pixels = if (channels == 3) {
        val transposedFloatBuffer = FloatArray(floatBuffer.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                for (c in 0 until channels) {
                    val sourceIndex = c * height * width + y * width + x
                    val targetIndex =
                        y * width * channels + x * channels + (2 - c) // Swap R and B channels
                    transposedFloatBuffer[targetIndex] = floatBuffer[sourceIndex]
                }
            }
        }
        transposedFloatBuffer
    } else {
        floatBuffer
    }

    // Create a bitmap from the tensor data
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

    // Populate the bitmap with tensor data
    val pixelValues = IntArray(width * height)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val offset = y * width * channels + x * channels
            val r = (pixels[offset] * whiteBalance[0]).coerceIn(0f, 1f)
                .gammaCorrect(gamma) // Applying white balance and gamma correction to each channel
            val g = (pixels[offset + 1] * whiteBalance[1]).coerceIn(0f, 1f).gammaCorrect(gamma)
            val b = (pixels[offset + 2] * whiteBalance[2]).coerceIn(0f, 1f).gammaCorrect(gamma)
            pixelValues[y * width + x] = Color.rgb(
                (b * 255).toInt(),
                (g * 255).toInt(),
                (r * 255).toInt()
            ) // Swap R and B channels
        }
    }

    bitmap.setPixels(pixelValues, 0, width, 0, 0, width, height)

    return bitmap
}

// Extension function for gamma correction
private fun Float.gammaCorrect(gamma: Float): Float {
    return this.pow(gamma)
}

private fun saveMediaToStorage(bitmap: Bitmap, context: Context) {
    //Generating a file name
    val filename = "${System.currentTimeMillis()}.png"

    //Output stream
    var fos: OutputStream? = null

    //For devices running android >= Q
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        //getting the contentResolver
        context?.contentResolver?.also { resolver ->

            //Content resolver will process the contentvalues
            val contentValues = ContentValues().apply {

                //putting file information in content values
                put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                put(MediaStore.MediaColumns.MIME_TYPE, "image/jpg")
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES)
            }

            //Inserting the contentValues to contentResolver and getting the Uri
            val imageUri: Uri? =
                resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            //Opening an outputstream with the Uri that we got
            fos = imageUri?.let { resolver.openOutputStream(it) }
        }
    } else {
        //These for devices running on android < Q
        //So I don't think an explanation is needed here
        val imagesDir =
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val image = File(imagesDir, filename)
        fos = FileOutputStream(image)
    }

    fos?.use {
        //Finally writing the bitmap to the output stream that we opened
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, it)
    }
}

private fun preProcessing(bitmap: Bitmap): Tensor {
    val width = bitmap.width
    val height = bitmap.height

    //Creating float32 array for holding bitmap rgb colors
    val inputData = FloatArray(width * height * 3)

    //Getting bitmap pixels in IntArray
    val intValues = IntArray(width * height)
    bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

    for (i in 0 until height * width) {
        val pixel = intValues[i]
        // Extract RGB components and normalize
        val red = ((pixel shr 16) and 0xFF) / 255.0f
        val green = ((pixel shr 8) and 0xFF) / 255.0f
        val blue = (pixel and 0xFF) / 255.0f

        // Store RGB components into float array
        inputData[i] = red
        inputData[i + width * height] = green
        inputData[i + 2 * width * height] = blue
    }

    // Convert float array to tensor
    val tensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, height.toLong(), width.toLong()))
    return tensor
}

private fun postProcessing(tensor: Tensor): Bitmap {
    val height = tensor.shape()[2].toInt()
    val width = tensor.shape()[3].toInt()

    // Create empty bitmap in RGB format
    val bmp: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565)
    val pixels = IntArray(width * height)

    // Convert tensor to float array
    val floatArray = tensor.dataAsFloatArray

    // mapping smallest value to 0 and largest value to 255
    val maxValue = floatArray.max()
    val minValue = floatArray.min()
    val delta = maxValue - minValue

    // Define if float min..max will be mapped to 0..255 or 255..0
    val conversion = { v: Float -> ((v - minValue) / delta * 255).roundToInt() }

    // copy each value from float array to RGB channels
    for (i in 0 until width * height) {
        val r = conversion(floatArray[i])
        val g = conversion(floatArray[i + width * height])
        val b = conversion(floatArray[i + 2 * width * height])
        // set pixel colors
        pixels[i] = Color.rgb(r, g, b)
    }

    // set bitmap pixels
    bmp.setPixels(pixels, 0, width, 0, 0, width, height)

    return bmp

}

@Throws(IOException::class)
fun assetFilePath(context: Context, assetName: String?): String? {
    val file = assetName?.let { File(context.filesDir, it) }!!
    if (file.exists() && file.length() > 0) {
        return file.absolutePath
    }
    context.assets.open(assetName).use { `is` ->
        FileOutputStream(file).use { os ->
            val buffer = ByteArray(4 * 1024)
            var read: Int
            while (`is`.read(buffer).also { read = it } != -1) {
                os.write(buffer, 0, read)
            }
            os.flush()
        }
        return file.absolutePath
    }
}

fun whiteBalanceTransform(bitmap: Bitmap): Bitmap {

    val bmp: Bitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.RGB_565)

    var sumRed = 0
    var sumGreen = 0
    var sumBlue = 0
    var totalPixels = 0

    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)

            sumRed += red
            sumGreen += green
            sumBlue += blue
            totalPixels++
        }
    }

    val avgRed = sumRed / totalPixels
    val avgGreen = sumGreen / totalPixels
    val avgBlue = sumBlue / totalPixels

    val maxAvg = maxOf(avgRed, avgGreen, avgBlue).toDouble()
    val redFactor = maxAvg / avgRed
    val greenFactor = maxAvg / avgGreen
    val blueFactor = maxAvg / avgBlue

    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val red = (Color.red(pixel) * redFactor).toInt().coerceIn(0..255)
            val green = (Color.green(pixel) * greenFactor).toInt().coerceIn(0..255)
            val blue = (Color.blue(pixel) * blueFactor).toInt().coerceIn(0..255)
            val newPixel = Color.rgb(red, green, blue)
            bmp.setPixel(x, y, newPixel)
        }
    }

    return bmp
}

fun gammaCorrection(bitmap: Bitmap, gamma: Float): Bitmap {

    val bmp: Bitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.RGB_565)

    val gammaTable = IntArray(256)
    for (i in gammaTable.indices) {
        val value = (i / 255.0f).pow(1.0f / gamma) * 255.0f
        gammaTable[i] = value.toInt().coerceIn(0..255)
    }

    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val red = gammaTable[Color.red(pixel)]
            val green = gammaTable[Color.green(pixel)]
            val blue = gammaTable[Color.blue(pixel)]
            val newPixel = Color.rgb(red, green, blue)
            bmp.setPixel(x, y, newPixel)
        }
    }

    return bmp
}

fun histogramEqualization(bitmap: Bitmap): Bitmap {

    val bmp: Bitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Bitmap.Config.RGB_565)

    val histogram = IntArray(256)
    val cdf = IntArray(256)
    val totalPixels = bitmap.width * bitmap.height

    // Calculate the histogram
    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val intensity = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            histogram[intensity]++
        }
    }

    // Calculate the cumulative distribution function (CDF)
    cdf[0] = histogram[0]
    for (i in 1 until 256) {
        cdf[i] = cdf[i - 1] + histogram[i]
    }

    // Calculate the equalized pixel values
    val equalizedHistogram = IntArray(256)
    for (i in 0 until 256) {
        equalizedHistogram[i] = (cdf[i] * 255 / totalPixels)
    }

    // Apply histogram equalization
    for (x in 0 until bitmap.width) {
        for (y in 0 until bitmap.height) {
            val pixel = bitmap.getPixel(x, y)
            val intensity = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            val equalizedIntensity = equalizedHistogram[intensity]
            val newPixel = Color.rgb(equalizedIntensity, equalizedIntensity, equalizedIntensity)
            bmp.setPixel(x, y, newPixel)
        }
    }

    return bmp
}

fun bitmapToRgbNorm(bitmap: Bitmap): FloatArray {
    val width = bitmap.width
    val height = bitmap.height

    val inputData = FloatArray(width * height * 3)

    val intValues = IntArray(width * height)
    bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

    for (i in 0 until height * width) {
        val pixel = intValues[i]
        // Extract RGB components and normalize
        val red = Color.red(pixel) / 255.0f
        val green = Color.green(pixel) / 255.0f
        val blue = Color.blue(pixel) / 255.0f

        inputData[i] = red
        inputData[i + width * height] = green
        inputData[i + 2 * width * height] = blue
    }
    return inputData
}

fun divideBitmap(bitmap: Bitmap, maxRow: Int, maxColum: Int): List<Bitmap> {
    val subImages = mutableListOf<Bitmap>()
    val subimageWidth = 128
    val subimageHeight = 128
    val totalWidth = bitmap.width
    val totalHeight = bitmap.height

    for (i in 0 until maxColum) {
        for (j in 0 until maxRow) {
            val startX = j * subimageWidth
            val startY = i * subimageHeight
            val endX =
                if (startX + subimageWidth > totalWidth) totalWidth else startX + subimageWidth
            val endY =
                if (startY + subimageHeight > totalHeight) totalHeight else startY + subimageHeight
            val subImage =
                Bitmap.createBitmap(endX - startX, endY - startY, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(subImage)
            val srcRect = Rect(startX, startY, endX, endY)
            val destRect = Rect(0, 0, endX - startX, endY - startY)
            canvas.drawBitmap(bitmap, srcRect, destRect, null)
            subImages.add(subImage)
        }
    }
    return subImages
}


fun removePadding(paddedBitmap: Bitmap, paddingWidth: Int, paddingHeight: Int): Bitmap {
    val paddedWidth = paddedBitmap.width
    val paddedHeight = paddedBitmap.height

    val newWidth = paddedWidth - paddingWidth
    val newHeight = paddedHeight - paddingHeight

    val croppedBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(croppedBitmap)

    // Calculate the source rectangle for the padded bitmap
    val left = paddingWidth / 2
    val top = paddingHeight / 2
    val right = left + newWidth
    val bottom = top + newHeight
    val srcRect = Rect(left, top, right, bottom)

    // Draw the padded bitmap onto the cropped bitmap
    canvas.drawBitmap(paddedBitmap, srcRect, Rect(0, 0, newWidth, newHeight), null)

    return croppedBitmap
}

private fun removePaddingRG(paddedBitmap: Bitmap, paddingWidth: Int, paddingHeight: Int): Bitmap {
    val paddedWidth = paddedBitmap.width
    val paddedHeight = paddedBitmap.height

    val newWidth = paddedWidth - paddingWidth
    val newHeight = paddedHeight - paddingHeight

    val croppedBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(croppedBitmap)

    // Calculate the source rectangle for the padded bitmap
    val left = paddingWidth
    val top = paddingHeight
    val right = left + newWidth
    val bottom = top + newHeight
    val srcRect = Rect(left, top, right, bottom)

    // Draw the padded bitmap onto the cropped bitmap
    canvas.drawBitmap(paddedBitmap, srcRect, Rect(0, 0, newWidth, newHeight), null)

    return croppedBitmap
}


private fun findNearestSize(
    inputWidth: Int,
    inputHeight: Int,
    availableSizes: List<ImageSize>
): Pair<Boolean, ImageSize> {
    // Check if the input size matches the first available size exactly
    if (inputWidth == availableSizes.first().width && inputHeight == availableSizes.first().height) {
        return false to availableSizes.first()
    }

    // Iterate through the available sizes to find the next nearest size
    for (size in availableSizes) {
        if (inputWidth <= size.width && inputHeight <= size.height) {
            return true to size
        }
    }

    // If no suitable size is found, return the first size by default (shouldn't happen with valid inputs)
    return true to availableSizes.first()
}


private fun calculatePadding(inputSize: ImageSize, nearestSize: ImageSize): Pair<Int, Int> {
    val paddingWidth = nearestSize.width - inputSize.width
    val paddingHeight = nearestSize.height - inputSize.height
    return Pair(paddingWidth, paddingHeight)
}

private fun addPadding(originalBitmap: Bitmap, paddingWidth: Int, paddingHeight: Int): Bitmap {
    val originalWidth = originalBitmap.width
    val originalHeight = originalBitmap.height

    val newWidth = originalWidth + paddingWidth
    val newHeight = originalHeight + paddingHeight

    val paddedBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(paddedBitmap)

    // Clear the canvas with a default background color (e.g., white)
    canvas.drawColor(android.graphics.Color.WHITE)

    // Calculate the destination rectangle for the original bitmap
    val left = paddingWidth
    val top = paddingHeight
    val right = left + originalWidth
    val bottom = top + originalHeight
    val destRect = Rect(left, top, right, bottom)

    // Draw the original bitmap onto the padded bitmap
    canvas.drawBitmap(originalBitmap, null, destRect, null)

    return paddedBitmap
}

