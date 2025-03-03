//シリアルモニタによる制御

#include <Arduino_FreeRTOS.h>
#include <Wire.h>


void TaskA( void *pvParameters );
void TaskB( void *pvParameters );
void TaskC( void *pvParameters );

int tDelay = 100;   //100ms遅延設定
int PWMPin = 10;   // PWM cntrol


//const float Kp=100.0;//pid control parameter
//const float Ki=0.0;
//const float Kd=0.1;

float Target;
float TargetH;
float TargetL;

const float delta=0.5;

float x;
//float dt;
//float pretime;
//float P, I, D, preP;

uint16_t duty = 0;

uint8_t Log = 0;

uint8_t sw_on;

unsigned long startTime, stopTime; // 開始時刻と終了時刻

String str;


void setup() 
{
  pinMode(PWMPin, OUTPUT);   //10番ピンをOUTPUTとして定義
  Wire.begin();


  // Now set up two tasks to run independently.
  xTaskCreate(
    TaskA
    ,  "TaskA"   // A name just for humans
    ,  128  // This stack size can be checked & adjusted by reading the Stack Highwater
    ,  NULL
    ,  1  // Priority, with 3 (configMAX_PRIORITIES - 1) being the highest, and 0 being the lowest.
    ,  NULL );

  xTaskCreate(
    TaskB
    ,  "TaskB"
    ,  128  // Stack size
    ,  NULL
    ,  2  // Priority
    ,  NULL );

  xTaskCreate(
    TaskC
    ,  "TaskC"
    ,  128  // Stack size
    ,  NULL
    ,  3  // Priority
    ,  NULL );

  Serial.begin(9600);         //シリアル通信のデータ転送レート設定しポート開放
}

void loop()
{
  // Empty. Things are done in Tasks.
}

void TaskA(void *pvParameters)  // This is a task.
{
  (void) pvParameters;

  for (;;) // A Task shall never return or exit.
  {
    analogWrite(PWMPin, (uint8_t)duty);//pwn control duty = 0 ~ 255 : 0 ~ 100%

    vTaskDelay( tDelay / portTICK_PERIOD_MS ); //wait 100ms

    Wire.beginTransmission(0x3D); //thermopile i2c Slave address
    Wire.write(0x71); //thermopile i2c Object temperature register
    Wire.endTransmission(false);
    Wire.requestFrom(0x3D, 2, true);
    int val_L = Wire.read();
    int val_H = Wire.read();
    Wire.endTransmission();
    x=(float)(word(val_H,val_L))/8-30; //Thermopile register load

    if(sw_on){
      if(x>TargetH){ //温度が高い場合dutyをオフ
        duty = 0;
      }else if(x<=TargetL){//温度が低い場合dutyをオン
        duty = 255;
      }else{
        ;
      }
    }

  }
}

void TaskB(void *pvParameters)  // This is a task.
{
  (void) pvParameters;

  for (;;)
  {
    if (Serial.available() > 0 )     //受信したデータが存在した場合以下を実行
    {
      String data = Serial.readStringUntil('\n');

      if (data[0] == 'w')
      {
        Serial.println("Temp logging starts"); //wコマンド：ログスタート
        Log = 1;
        startTime = millis();
      }
      else if (data[0] == 'x')
      {
        Serial.print("Start of "); //xコマンド：刺激開始
        Serial.print(data.substring(1)); //xコマンド：刺激開始
        Serial.println(" degree stimulus"); //xコマンド：刺激開始
        int d10 = data[1]- 0x30;
        int d1 = data[2]- 0x30;
        Target = (float) d10 * 10 + d1;
        if(Target <= 20.0)Target=20.0;
        else if(Target > 60.0)Target=60.0;
        TargetH = Target + delta;
        TargetL = Target - delta;
        sw_on = 1;
      }
      else if (data[0] == 'y')
      {
        Serial.println("End of stimulus"); //yコマンド：刺激終了
        Target = 0;
        TargetH = 0;
        TargetL = 0;
        sw_on = 0;
        duty = 0;
      }
      else if (data[0] == 'z')
      {
        Serial.println("Temp logging ends"); //zコマンド：ログ終了
        Log = 0;
      }

    }

    if(Log)
    {
      stopTime = millis();
      str = "Time:"+String(stopTime - startTime) + ",Temp:"+String(x,2);
      Serial.println(str);
    }

    vTaskDelay( 1000 / portTICK_PERIOD_MS ); // plot T=1000ms = 1s
  }
}

void TaskC(void *pvParameters)  // This is a task.
{
  (void) pvParameters;

  for (;;)
  {
    vTaskDelay( 120000 / portTICK_PERIOD_MS ); // 120000ms=120s=2min
  }
}


