package com.example.imageenhancer.screens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Camera
import android.graphics.Matrix
import android.provider.MediaStore
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraEffect
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProcessor
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsEndWidth
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cameraswitch
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.imageenhancer.viewModels.CameraScreenViewModel
import java.util.function.Consumer
import kotlin.math.abs

@Composable
fun CameraScreen(onEnhanceClick: (Bitmap, Int, Int) -> Unit) {

    val context = LocalContext.current

    val controller = remember {
        LifecycleCameraController(context).apply {
            setEnabledUseCases(CameraController.IMAGE_CAPTURE)
        }
    }

    var bitmap: Bitmap

    val viewModel: CameraScreenViewModel = hiltViewModel()
    var isSwitchChecked by remember {
        mutableStateOf(false)
    }

    val launcher =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.OpenDocument()) {
            it?.let { uri ->
                bitmap = MediaStore.Images.Media.getBitmap(
                    context.contentResolver,
                    uri
                )
                /*val finalBitmap: Bitmap
                if (isSwitchChecked) {
                    finalBitmap = viewModel.getThreeByFourBitmap(bitmap)
                } else {
                    finalBitmap = viewModel.getOneByOneBitmap(bitmap)
                }*/
                val paddingHeight = abs(768 - bitmap.height)

                val paddingWidth = abs(1024 - bitmap.width)

                val finalBitmap = viewModel.addPadding(bitmap, paddingWidth, paddingHeight)

//                Bitmap.createBitmap(bitmap, 0, 0, 375, 500, null, false)

                Log.d("imageBitmap", "onCreate: $finalBitmap")
                onEnhanceClick(finalBitmap, paddingWidth, paddingHeight)

            }
        }



    Box(contentAlignment = Alignment.Center, modifier = Modifier.fillMaxSize()) {

        CameraPreviewScreen(controller = controller)

        Column(
            modifier = Modifier
                .fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            /*Text(
                text = "Aspect Ratio", modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 20.dp, top = 20.dp),
                color = MaterialTheme.colorScheme.secondary
            )
            Row(
                verticalAlignment = Alignment.CenterVertically, modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 20.dp, start = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text(
                    text = "1/1",
                    color = if (!isSwitchChecked) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onPrimary
                )
                Switch(
                    checked = isSwitchChecked,
                    onCheckedChange = { isSwitchChecked = !isSwitchChecked },
                    colors = SwitchDefaults.colors(
                        uncheckedIconColor = MaterialTheme.colorScheme.primary,
                        uncheckedThumbColor = MaterialTheme.colorScheme.primary,
                        uncheckedBorderColor = MaterialTheme.colorScheme.primary,
                        uncheckedTrackColor = MaterialTheme.colorScheme.onPrimary
                    )
                )
                Text(
                    text = "3/4",
                    color = if (!isSwitchChecked) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.primary
                )
            }*/

            Spacer(modifier = Modifier.weight(1f))

            Row(
                modifier = Modifier
                    .background(color = Color.Black.copy(0.25f))
                    .height(150.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround,
                verticalAlignment = Alignment.CenterVertically
            ) {

                IconButton(onClick = {
                    controller.cameraSelector =
                        if (controller.cameraSelector == CameraSelector.DEFAULT_BACK_CAMERA) {
                            CameraSelector.DEFAULT_FRONT_CAMERA
                        } else {
                            CameraSelector.DEFAULT_BACK_CAMERA
                        }

                }) {

                    Icon(
                        modifier = Modifier.size(30.dp),
                        imageVector = Icons.Default.Cameraswitch,
                        contentDescription = "switch Camera"
                    )

                }

                IconButton(
                    onClick = {
                        takePhoto(
                            context = context,
                            controller = controller,
                            viewModel = viewModel,
                            isSwitchChecked = isSwitchChecked
                        ) { bitmap, paddingWidth, paddingHeight ->
//                            Log.d("imageBitmap", "onCreate: $it")
                            onEnhanceClick(bitmap, paddingWidth, paddingHeight)
                        }
                    },
                    modifier = Modifier
                        .size(100.dp)
                ) {

                    Icon(
                        imageVector = Icons.Default.Circle,
                        contentDescription = "",
                        modifier = Modifier.size(100.dp),
                        tint = Color.White.copy(alpha = 0.5f)
                    )

                    Icon(
                        imageVector = Icons.Default.Circle,
                        modifier = Modifier.size(80.dp),
                        contentDescription = "Take Picture"
                    )
                }

                IconButton(onClick = {
                    launcher.launch(arrayOf("image/*"))
                }) {
                    Icon(
                        modifier = Modifier.size(30.dp),
                        imageVector = Icons.Default.PhotoLibrary,
                        contentDescription = "Open Gallery"
                    )
                }

            }
        }


    }
}


private fun takePhoto(
    context: Context,
    controller: LifecycleCameraController,
    viewModel: CameraScreenViewModel,
    isSwitchChecked: Boolean,
    onPhotoTaken: (bitmap: Bitmap, paddingWidth: Int, paddingHeight: Int) -> Unit
) {

    controller.takePicture(
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                super.onCaptureSuccess(image)

                val rotation = image.imageInfo.rotationDegrees.toFloat()
                Log.d("imageBitmap", "onCaptureSuccess: $rotation")

                val matrix: Matrix = Matrix()

                var bmp: Bitmap = image.toBitmap()

                if (rotation == 90f || rotation == 270f) {
                    val newMatrix: Matrix = Matrix()
                    newMatrix.preRotate(if (90f == rotation) rotation else -90f)

                    bmp =
                        Bitmap.createBitmap(
                            image.toBitmap(),
                            0,
                            0,
                            image.toBitmap().width,
                            image.toBitmap().height,
                            newMatrix,
                            false
                        )

                } else
                    matrix.preRotate(rotation)

                onPhotoTaken(
                    /*if (!isSwitchChecked) viewModel.getOneByOneBitmap(
                        bmp,
                        matrix
                    ) else viewModel.getThreeByFourBitmap(bmp, matrix)*/

                    viewModel.addPadding(bmp, abs(1024 - bmp.width), abs(768 - bmp.height)),
                    abs(1024 - bmp.width),
                    abs(768 - bmp.height)
                )

            }

            override fun onError(exception: ImageCaptureException) {
                super.onError(exception)
                exception.printStackTrace()
            }
        }
    )

}