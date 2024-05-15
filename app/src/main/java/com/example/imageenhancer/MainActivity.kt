package com.example.imageenhancer

import android.content.ComponentCallbacks2
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.imageenhancer.screens.CameraScreen
import com.example.imageenhancer.screens.EnhanceScreen
import com.example.imageenhancer.ui.theme.ImageEnhancerTheme
import dagger.hilt.android.AndroidEntryPoint
import java.io.File
import java.io.FileOutputStream

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (!hasRequiredPermissions()) {
            ActivityCompat.requestPermissions(
                this,
                CAMERA_PERMISSION,
                0
            )
        }

        onTrimMemory(level = ComponentCallbacks2.TRIM_MEMORY_COMPLETE)

        setContent {
            ImageEnhancerTheme {

                val context = LocalContext.current

                val navController = rememberNavController()

                Scaffold {
                    NavHost(
                        navController = navController,
                        startDestination = "cameraScreen",
                        modifier = Modifier.padding(it)
                    ) {

                        composable("cameraScreen") {
                            CameraScreen { bitmap ->
                                val path = context.getExternalFilesDir(null)!!.absolutePath
                                val tempFile = File(path, "tempFileName.jpg")
                                val fOut = FileOutputStream(tempFile)
                                bitmap.compress(Bitmap.CompressFormat.WEBP_LOSSLESS, 50, fOut)
                                fOut.close()
                                navController.navigate("enhanceScreen")
                            }
                        }

                        composable("enhanceScreen") {
                            val path = context.getExternalFilesDir(null)!!.absolutePath
                            val imagePath = "$path/tempFileName.jpg"

                            val image = BitmapFactory.decodeFile(imagePath)
                            File(imagePath).deleteOnExit() // Delete temp image

                            EnhanceScreen(image) {
                                navController.popBackStack()
                            }
                        }

                    }
                }
            }
        }
    }


    private fun hasRequiredPermissions(): Boolean {
        return CAMERA_PERMISSION.all {
            checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED
        }
    }

    companion object {
        private val CAMERA_PERMISSION = arrayOf(
            android.Manifest.permission.CAMERA
        )
    }


}

