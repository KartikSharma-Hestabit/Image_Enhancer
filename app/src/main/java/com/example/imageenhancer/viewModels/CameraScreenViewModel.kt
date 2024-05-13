package com.example.imageenhancer.viewModels

import android.graphics.Bitmap
import android.graphics.Matrix
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

            return Bitmap.createScaledBitmap(bmp, 800, 600, false)

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

            return Bitmap.createScaledBitmap(bmp, 600, 800, false)
        }

    }


    fun getOneByOneBitmap(bitmap: Bitmap, matrix: Matrix = Matrix()): Bitmap {

        val bmp: Bitmap

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

            return Bitmap.createScaledBitmap(bmp, 1000, 1000, false)

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

            return Bitmap.createScaledBitmap(bmp, 1000, 1000, false)
        }

    }

}