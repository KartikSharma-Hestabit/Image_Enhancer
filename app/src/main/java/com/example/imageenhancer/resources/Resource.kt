package com.example.imageenhancer.resources

import java.lang.Exception

sealed class Resource<out R> {
    data class Success<out R>(val result: R) : Resource<R>()
    data class Failure(val exception: Exception) : Resource<Nothing>()
    data class Loading(val message: String) : Resource<Nothing>()
}