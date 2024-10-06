# Generated by Django 4.2.16 on 2024-10-03 15:51

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('trading_app', '0003_remove_assethistory_asset_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='assethistory',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='candles', to='trading_app.dataset'),
        ),
        migrations.CreateModel(
            name='TrainingSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('long', models.BooleanField(default=False)),
                ('short', models.BooleanField(default=False)),
                ('stop_loss', models.FloatField(default=0.0)),
                ('indicators', models.JSONField(default=list)),
                ('epochs', models.IntegerField(default=10)),
                ('batch_size', models.IntegerField(default=32)),
                ('learning_rate', models.FloatField(default=0.001)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=20)),
                ('progress', models.FloatField(default=0.0)),
                ('accuracy', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('current_epoch', models.IntegerField(default=0)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trading_app.dataset')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trading_app.userprofile')),
            ],
        ),
        migrations.CreateModel(
            name='TrainedModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('model_file', models.FileField(upload_to='models/')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('training_session', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='trading_app.trainingsession')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='trading_app.userprofile')),
            ],
        ),
    ]