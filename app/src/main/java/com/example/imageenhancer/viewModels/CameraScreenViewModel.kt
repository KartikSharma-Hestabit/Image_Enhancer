package com.example.imageenhancer.viewModels

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import androidx.lifecycle.ViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class CameraScreenViewModel @Inject constructor() : ViewModel() {

    fun getThreeByFourBitmap(bitmap: Bitmap, matrix: Matrix = Matrix()): Bitmap {

        if (bitmap.width == bitmap.height) return bitmap

        val bmp: Bitmap

        if (bitmap.width >= bitmap.height) {

            matrix.preScale(4f, 3f)

            val widthOffset = (bitmap.width / 4 - bitmap.height / 3) * 2
            val actualWidth = bitmap.width - (widthOffset * 2)

            bmp = Bitmap.createBitmap(
                bitmap,
                widthOffset,
                0,
                actualWidth,
                bitmap.height,
                matrix,
                false
            );

            return Bitmap.createScaledBitmap(bmp, 1024, 768, false)

        } else {

            matrix.preScale(3f, 4f)

            val heightOffset = (bitmap.height / 4 - bitmap.width / 3) * 2
            val actualHeight = bitmap.height - (heightOffset * 2)

            bmp = Bitmap.createBitmap(
                bitmap,
                0,
                heightOffset,
                bitmap.width,
                actualHeight,
                matrix,
                false

            )

            return Bitmap.createScaledBitmap(bmp, 768, 1024, false)
        }

    }


    fun getOneByOneBitmap(bitmap: Bitmap, matrix: Matrix = Matrix()): Bitmap {

        val bmp: Bitmap

        val imgDimension = 768

        matrix.preScale(1f, 1f)

        if (bitmap.width >= bitmap.height) {

            bmp = Bitmap.createBitmap(
                bitmap,
                bitmap.getWidth() / 2 - bitmap.getHeight() / 2,
                0,
                bitmap.getHeight(),
                bitmap.getHeight(),
                matrix,
                false
            )

            return Bitmap.createScaledBitmap(bmp, imgDimension, imgDimension, false)

        } else {

            bmp = Bitmap.createBitmap(
                bitmap,
                0,
                bitmap.getHeight() / 2 - bitmap.getWidth() / 2,
                bitmap.getWidth(),
                bitmap.getWidth(),
                matrix,
                false
            )

            return Bitmap.createScaledBitmap(bmp, imgDimension, imgDimension, false)
        }

    }

    fun addPadding(originalBitmap: Bitmap, paddingWidth: Int, paddingHeight: Int): Bitmap {
        Log.d("imageBitmap", "onCaptureSuccess: ${originalBitmap.width}, ${originalBitmap.height}")
        val originalWidth = originalBitmap.width
        val originalHeight = originalBitmap.height
        val newWidth = originalWidth + paddingWidth
        val newHeight = originalHeight + paddingHeight
        val paddedBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(paddedBitmap)
        // Clear the canvas with a default background color (e.g., white)
        canvas.drawColor(android.graphics.Color.WHITE)
        // Calculate the destination rectangle for the original bitmap
        val left = paddingWidth / 2
        val top = paddingHeight / 2
        val right = left + originalWidth
        val bottom = top + originalHeight
        val destRect = Rect(left, top, right, bottom)
        // Draw the original bitmap onto the padded bitmap
        canvas.drawBitmap(originalBitmap, null, destRect, null)
        Log.d("imageBitmap", "onCaptureSuccess: ${paddedBitmap.width}, ${paddedBitmap.height}")
        return paddedBitmap
    }

}