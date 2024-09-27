from django.contrib import admin
from .models import UserProfile, UserAdditionalInfo, AssetHistory

admin.site.register(UserProfile)
admin.site.register(UserAdditionalInfo)
admin.site.register(AssetHistory)
