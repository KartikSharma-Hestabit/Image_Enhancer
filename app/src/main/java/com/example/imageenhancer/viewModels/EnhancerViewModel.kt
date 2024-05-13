package com.example.imageenhancer.viewModels

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.imageenhancer.resources.Resource
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.Device
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.Calendar
import javax.inject.Inject
import kotlin.math.pow
import kotlin.math.roundToInt


@HiltViewModel
class EnhancerViewModel @Inject constructor() : ViewModel() {


    private val _moduleFlow = MutableStateFlow<Resource<Bitmap>?>(null)
    val moduleFlow: StateFlow<Resource<Bitmap>?> = _moduleFlow

    @RequiresApi(Build.VERSION_CODES.O)
    suspend fun waternet(context: Context, bitmap: Bitmap) = withContext(Dispatchers.IO) {

        _moduleFlow.value = Resource.Loading("Working on Color Correction...")

        Log.d(
            "moduleStats",
            "Waternet started: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                Calendar.getInstance().get(Calendar.SECOND)
            }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
        )

        val model: Module = LiteModuleLoader.load(assetFilePath(context, "waternet.ptl"))
        val shape = longArrayOf(1, 3, bitmap.height.toLong(), bitmap.width.toLong())

        val wbBitmap = whiteBalanceTransform(bitmap)
        val gammaBitmap = gammaCorrection(bitmap, 0.7f)
        val heBitmap = histogramEqualization(bitmap)

        val rgbTensor = Tensor.fromBlob(bitmapToRgbNorm(bitmap), shape)
        val wbTensor = Tensor.fromBlob(bitmapToRgbNorm(wbBitmap), shape)
        val gcTensor = Tensor.fromBlob(bitmapToRgbNorm(gammaBitmap), shape)
        val heTensor = Tensor.fromBlob(bitmapToRgbNorm(heBitmap), shape)
        val output = model
            .forward(
                IValue.from(rgbTensor),
                IValue.from(wbTensor),
                IValue.from(gcTensor),
                IValue.from(heTensor)
            )
            .toTensor()

        model.destroy()
//        swinIR(context, output)


        scuNet(context, output)

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

    suspend fun scuNet(context: Context, inputTensor: Tensor) = withContext(Dispatchers.IO) {

        _moduleFlow.value = Resource.Loading("Working on Denoising...")

        val model = LiteModuleLoader.load(assetFilePath(context, "scunet.ptl"))
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        val outputBitmap = tensorToBitmap(outputTensor)
        model.destroy()
        _moduleFlow.value = Resource.Success(outputBitmap)
        saveMediaToStorage(outputBitmap, context)

        Log.d(
            "moduleStats",
            "scuNet Completed: ${Calendar.getInstance().get(Calendar.MINUTE)}:${
                Calendar.getInstance().get(Calendar.SECOND)
            }:${Calendar.getInstance().get(Calendar.MILLISECOND)}"
        )

    }

}

fun tensorToBitmap(
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
fun Float.gammaCorrect(gamma: Float): Float {
    return this.pow(gamma)
}

fun saveMediaToStorage(bitmap: Bitmap, context: Context) {
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