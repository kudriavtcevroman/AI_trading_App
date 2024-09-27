from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib import messages

# Регистрация пользователя
def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            messages.success(request, f"Аккаунт {username} был успешно создан.")
            login(request, user)
            return redirect("home")  # Замените "home" на ваше целевое представление после входа
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})

# Вход в систему
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("home")  # Замените "home" на ваше целевое представление после входа
            else:
                messages.error(request, "Неправильное имя пользователя или пароль.")
        else:
            messages.error(request, "Неправильное имя пользователя или пароль.")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

# Выход из системы
def logout_view(request):
    logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("login")
