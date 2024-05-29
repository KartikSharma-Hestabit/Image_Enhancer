package com.swinir

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect

fun divideBitmap(bitmap: Bitmap,maxRowColum:Int): List<Bitmap> {
    val subImages = mutableListOf<Bitmap>()

    val subimageWidth = 128
    val subimageHeight = 128

    val totalWidth = bitmap.width
    val totalHeight = bitmap.height


    for (i in 0 until maxRowColum) {
        for (j in 0 until maxRowColum) {
            val startX = j * subimageWidth
            val startY = i * subimageHeight

            val endX = if (startX + subimageWidth > totalWidth) totalWidth else startX + subimageWidth
            val endY = if (startY + subimageHeight > totalHeight) totalHeight else startY + subimageHeight

            val subImage = Bitmap.createBitmap(endX - startX, endY - startY, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(subImage)
            val srcRect = Rect(startX, startY, endX, endY)
            val destRect = Rect(0, 0, endX - startX, endY - startY)
            canvas.drawBitmap(bitmap, srcRect, destRect, null)
            subImages.add(subImage)
        }
    }

    return subImages
}
