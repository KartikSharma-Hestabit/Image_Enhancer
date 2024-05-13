package com.example.imageenhancer.screens

import android.graphics.Bitmap
import android.os.Build
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.FilterBAndW
import androidx.compose.material.icons.filled.ImagesearchRoller
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import coil.compose.AsyncImage
import com.example.imageenhancer.resources.Resource
import com.example.imageenhancer.viewModels.EnhancerViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.concurrent.thread

@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun EnhanceScreen(bitmap: Bitmap, onRetake: () -> Unit) {

    val context = LocalContext.current
    val viewModel: EnhancerViewModel = hiltViewModel()
    val resultFlow = viewModel.moduleFlow.collectAsState()
    var isLoading by remember {
        mutableStateOf(false)
    }
    var message by remember {
        mutableStateOf("")
    }

    var enhancedBitmap: Bitmap by remember {
        mutableStateOf(bitmap)
    }

    var isEnhanced by remember {
        mutableStateOf(false)
    }

    var onEnhance by remember {
        mutableStateOf(false)
    }



    if (onEnhance)
        LaunchedEffect(key1 = Unit) {
            withContext(Dispatchers.IO) {
                viewModel.waternet(context, enhancedBitmap)
            }
        }

    resultFlow.value?.let {

        when (it) {
            is Resource.Failure -> {
                isLoading = false
                Toast.makeText(context, "${it.exception.message}", Toast.LENGTH_SHORT).show()
            }

            is Resource.Loading -> {
                isLoading = true
                message = it.message
            }

            is Resource.Success -> {
                isLoading = false
                isEnhanced = true
                enhancedBitmap = it.result
            }
        }

    }

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {

        val scale = remember { mutableStateOf(1f) }

        LaunchedEffect(key1 = Unit) {

            if (isEnhanced) Toast.makeText(context, "Saved to photos", Toast.LENGTH_SHORT).show()
        }

        Column(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.SpaceAround,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Enhance Your Image",
                fontWeight = FontWeight.Bold,
                fontSize = 25.sp,
                modifier = Modifier.padding(vertical = 20.dp)
            )

            Box(
                modifier = Modifier
                    .padding(10.dp)
                    .clip(RoundedCornerShape(5.dp)) // Clip the box content
                    .wrapContentSize() // Give the size you want...
                    .background(Color.Gray)
                    .pointerInput(Unit) {
                        detectTransformGestures { centroid, pan, zoom, rotation ->
                            scale.value *= zoom
                        }
                    }
            ) {
                AsyncImage(
                    model = enhancedBitmap,
                    contentDescription = "",
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.Center) // keep the image centralized into the Box
                        .graphicsLayer(
                            // adding some zoom limits (min 50%, max 200%)
                            scaleX = maxOf(1f, minOf(5f, scale.value)),
                            scaleY = maxOf(1f, minOf(5f, scale.value)),
                        ),
                    contentScale = ContentScale.FillWidth,
                )

            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround
            ) {
                TextButton(
                    onClick = { onRetake() },
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 10.dp)
                        .height(50.dp),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Icon(
                        imageVector = if (!isEnhanced) Icons.Default.Replay else Icons.Default.ImagesearchRoller,
                        contentDescription = "Retake Photo"
                    )
                    Spacer(modifier = Modifier.width(5.dp))
                    Text(text = if (!isEnhanced) "Retake Photo" else "Enhance more photos")
                }

                Spacer(modifier = Modifier.width(5.dp))

                if (!isEnhanced)
                    TextButton(
                        onClick = {
                            onEnhance = true
                        },
                        modifier = Modifier
                            .weight(1f)
                            .padding(end = 10.dp)
                            .height(50.dp),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.AutoAwesome,
                            contentDescription = "Enhance Image"
                        )
                        Spacer(modifier = Modifier.width(7.dp))
                        Text(text = "Enhance")
                    }
            }
        }


        if (isLoading) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(color = Color.Black.copy(0.5f)),
                contentAlignment = Alignment.Center
            ) {

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {


                    CircularProgressIndicator()

                    Spacer(modifier = Modifier.height(10.dp))

                    Text(
                        text = message,
                        color = Color.White,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }

    }
}