#include <Arduino.h>
#include <Servo.h>
#include <EEPROM.h>

Servo servo1;
Servo servo2;

String buf="";

const int SERVO1_PIN=3;
const int SERVO2_PIN=5;

/* PWM limits */

const int US_MIN=500;
const int US_MAX=2500;

/* calibration angles */

const int CAL_POINTS=4;
float cal_angles[CAL_POINTS]={0,45,90,135};

/* lookup tables (microseconds) */

float servo1_map[CAL_POINTS];
float servo2_map[CAL_POINTS];

enum Mode{
    MODE_IDLE,
    MODE_CAL,
    MODE_EXP
};

Mode mode=MODE_IDLE;

int cal_index=0;


/* clamp */

float clamp_us(float u){

    if(u<US_MIN) return US_MIN;
    if(u>US_MAX) return US_MAX;
    return u;
}



bool invalid_map(float *map) {

    for(int i=0;i<CAL_POINTS;i++){
        if(isnan(map[i]) || map[i] < US_MIN || map[i] > US_MAX)
            return true;
    }

    return false;
}

/* interpolation */

float interp(float theta,float *map){

    if(theta<=cal_angles[0]) return map[0];
    if(theta>=cal_angles[CAL_POINTS-1]) return map[CAL_POINTS-1];

    for(int i=0;i<CAL_POINTS-1;i++){

        float a=cal_angles[i];
        float b=cal_angles[i+1];

        if(theta>=a && theta<=b){

            float alpha=(theta-a)/(b-a);

            return map[i] + alpha*(map[i+1]-map[i]);
        }
    }

    return map[0];
}


/* move servos */

void move_servos(float d1,float d2){

    float us1=clamp_us(interp(d1,servo1_map));
    float us2=clamp_us(interp(d2,servo2_map));

    servo1.writeMicroseconds(us1);
    servo2.writeMicroseconds(us2);
}


/* calibration jogging */

void jog_servo(int id,float delta){

    if(id==1){

        servo1_map[cal_index]+=delta;
        servo1_map[cal_index]=clamp_us(servo1_map[cal_index]);

        servo1.writeMicroseconds(servo1_map[cal_index]);

        Serial.print("S1=");
        Serial.println(servo1_map[cal_index]);
    }

    if(id==2){

        servo2_map[cal_index]+=delta;
        servo2_map[cal_index]=clamp_us(servo2_map[cal_index]);

        servo2.writeMicroseconds(servo2_map[cal_index]);

        Serial.print("S2=");
        Serial.println(servo2_map[cal_index]);
    }
}


/* move to calibration target */

void goto_cal_point(){

    float target=cal_angles[cal_index];

    float us1=interp(target,servo1_map);
    float us2=interp(target,servo2_map);

    servo1.writeMicroseconds(us1);
    servo2.writeMicroseconds(us2);

    Serial.print("CAL ");
    Serial.println(target);
}


/* command parser */

void handle_command(String cmd){

    cmd.trim();

    if(cmd=="T" && mode==MODE_CAL){

        if(cal_index<CAL_POINTS-1) cal_index++;

        goto_cal_point();
        return;
    }

    if(cmd=="R" && mode==MODE_CAL){

        if(cal_index>0) cal_index--;

        goto_cal_point();
        return;
    }

    int c1=cmd.indexOf(',');
    String head=c1>0?cmd.substring(0,c1):cmd;

    if(head=="MODE"){

        String arg=cmd.substring(c1+1);

        if(arg=="CAL"){

            mode=MODE_CAL;
            cal_index=0;

            goto_cal_point();
        }

        if(arg=="IDLE"){

            mode=MODE_IDLE;
        }

        if(arg=="EXP"){

            mode=MODE_EXP;
        }

        return;
    }

    if(head=="CMD" && mode==MODE_EXP){

        int c2=cmd.indexOf(',',c1+1);

        float d1=cmd.substring(c1+1,c2).toFloat();
        float d2=cmd.substring(c2+1).toFloat();

        move_servos(d1,d2);

        return;
    }

    if(head=="JOG" && mode==MODE_CAL){

        int c2=cmd.indexOf(',',c1+1);

        int id=cmd.substring(c1+1,c2).toInt();
        float delta=cmd.substring(c2+1).toFloat();

        jog_servo(id,delta);

        return;
    }

    if(cmd=="SAVE"){

        EEPROM.put(0,servo1_map);
        EEPROM.put(sizeof(servo1_map),servo2_map);

        Serial.println("CAL SAVED");

        return;
    }
}


/* setup */

void setup(){

    Serial.begin(115200);

    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);

    EEPROM.get(0,servo1_map);
    EEPROM.get(sizeof(servo1_map),servo2_map);

    if(invalid_map(servo1_map) || invalid_map(servo2_map)){

        Serial.println("EEPROM invalid → initializing defaults");

        float defaults[CAL_POINTS] = {500, 1000, 1500, 2000};

        for(int i=0;i<CAL_POINTS;i++){
            servo1_map[i] = defaults[i];
            servo2_map[i] = defaults[i];
        }
    }

    servo1.writeMicroseconds(1500);
    servo2.writeMicroseconds(1500);

    Serial.println("READY");
}


/* main loop */

void loop(){

    while(Serial.available()){

        char c=Serial.read();

        if(c=='\r') continue;

        if(c=='\n'){

            handle_command(buf);
            buf="";
        }
        else buf+=c;
    }
}