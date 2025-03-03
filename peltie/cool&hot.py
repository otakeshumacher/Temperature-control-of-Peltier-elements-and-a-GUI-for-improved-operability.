import os
import time
import threading
import ctypes
import serial
import serial.tools.list_ports

from tkinter import *
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import filedialog

from obswebsocket import obsws, requests

from dotenv import load_dotenv
load_dotenv('.env')

INITIAL_TIME = 1000
RECORDING_TIME = 60000

Serial_Port = ''
LOG_DIR = r"C:\Users\otake\OneDrive - 東京理科大学 (1)\mause実験\ハンドリング20240903"  # LOG_DIRの定義
OBS_DIR = r"C:\Users\otake\OneDrive - 東京理科大学 (1)\mause実験\ハンドリング20240903"


def trigger(event):
    global Serial_Port
    serial_open()

    time.sleep(1)

    thread_2 = threading.Thread(target=serial_read, daemon = True)
    thread_2.start()

    #time.sleep(1)

    if obs1_check['state'] == 'enable' and obs1_cam.get():
        thread_1 = threading.Thread(target=Recording)
        thread_1.start()
    if obs2_check['state'] == 'enable' and obs2_cam.get():
        thread_3 = threading.Thread(target=Recording2)
        thread_3.start()
        
    t1 = INITIAL_TIME
    t2 = INITIAL_TIME + RECORDING_TIME
    if stm.get():
        t3 = int(stim_combo_2.get())*1000 + INITIAL_TIME + RECORDING_TIME
        t4 = int(stim_combo_2.get())*1000 + INITIAL_TIME + RECORDING_TIME * 2
        t5 = int(stim_combo_2.get())*1000 + INITIAL_TIME * 2 + RECORDING_TIME * 2
    else :
        t3 = INITIAL_TIME + RECORDING_TIME
        t4 = INITIAL_TIME + RECORDING_TIME * 2
        t5 = INITIAL_TIME * 2 + RECORDING_TIME * 2

    
    root.after(t1, command_w)#刺激開始 　ログ開始
    root.after(t1, command_x0)

    if stm.get():
        root.after(t2, command_x)#刺激開始 　ログ開始から 60000msec = 60sec後
        root.after(t3, command_x0)#刺激終了　ログ開始から 75sec後
    root.after(t4, command_z)#ログ終了　ログ開始から 135sec後
    root.after(t5, command_close)#ログ終了　ログ開始から 135sec後

def Recording():
    global ws
    obs_path = os.path.join(OBS_DIR, obs1_video_name_entry.get())

    ws.call(requests.SetProfileParameter(parameterCategory='SimpleOutput', parameterName='FilePath', parameterValue=obs_path))
    
    r1 = INITIAL_TIME
    if stm.get():
        r4 = int(stim_combo_2.get())*1000 + INITIAL_TIME + RECORDING_TIME * 2
    else :
        r4 = INITIAL_TIME + RECORDING_TIME * 2
    
    root.after(r1, rec_on)
    root.after(r4, rec_off)

def create_folder():
    base_path = OBS_DIR  # ここにベースディレクトリのパスを設定
    folder_name = entry.get()

    if not folder_name:
        messagebox.showwarning("Warning", "フォルダ名を入力してください")
        return

    new_folder_path = os.path.join(base_path, folder_name)

    if os.path.exists(new_folder_path):
        messagebox.showwarning("Warning", f"フォルダ '{folder_name}' は既に存在します")
    else:
        try:
            os.makedirs(new_folder_path)
            messagebox.showinfo("Info", f"フォルダ '{folder_name}' を作成しました")
        except Exception as e:
            messagebox.showerror("Error", f"フォルダの作成に失敗しました: {e}")

    
def Recording2():
    global ws2
    obs_path_2 = os.path.join(obs2_entry.get(), obs2_video_name_entry.get() + ".mkv")

    ws2.call(requests.SetProfileParameter(parameterCategory='SimpleOutput', parameterName='FilePath', parameterValue=obs_path_2))
    
    r1 = INITIAL_TIME
    if stm.get():
        r4 = int(stim_combo_2.get())*1000 + INITIAL_TIME + RECORDING_TIME * 2
    else :
        r4 = INITIAL_TIME + RECORDING_TIME * 2
    
    root.after(r1, rec_on_2)
    root.after(r4, rec_off_2)

def update_filename():
    # 名前を更新
    obs_video_name = obs1_video_name_entry.get()
    obs1_video_name_entry.delete(0, 'end')
    obs1_video_name_entry.insert(0, "")


def command_w():
    global Serial_Port
    Serial_Port.write('w\r\n'.encode('utf-8'))

def command_x0():
    global Serial_Port
    str1 = 'x'+'25'+'\r\n'
    Serial_Port.write(str1.encode('utf-8'))

def command_x():
    global Serial_Port
    str1 = 'x'+stim_combo.get()+'\r\n'
    Serial_Port.write(str1.encode('utf-8'))
    
def command_y():
    global Serial_Port
    Serial_Port.write('y\r\n'.encode('utf-8'))
    
def command_z():
    global Serial_Port
    Serial_Port.write('z\r\n'.encode('utf-8'))
    
def command_close():
    global Serial_Port
    Serial_Port.close()

def rec_on():
    global ws
    ws.call(requests.StartRecord())
    print("start")
    
def rec_off():
    global ws
    ws.call(requests.StopRecord())
    print("stop")
    
def rec_on_2():
    global ws2
    ws2.call(requests.StartRecord())
    print("start")
    
def rec_off_2():
    global ws2
    ws2.call(requests.StopRecord())
    print("stop")
    
def serial_read():
    global Serial_Port
    #0sec:  6,5sec:  4,10sec:  6,15sec:  5,20sec:  5,25sec:  5,30sec:  5
    #0sec:115,5sec:122,10sec:127,15sec:131,20sec:136,25sec:141,30sec:146
    #0sec:121,5sec:126,10sec:131,15sec:136,20sec:141,25sec:146,30sec:151
    n = stim_combo_2.current()
    reduce_list = [146, 141, 136, 131, 127, 122]
    nn = reduce_list[n]

    if not stm.get():
        nn = 115
    log_file_name = log_name_entry.get()
    log_file_path = os.path.join(LOG_DIR, log_name_entry.get())
    
    if not log_file_name:
        messagebox.showwarning("Warning", "ファイル名を入力してください")
        return
    
    if os.path.exists(log_file_name):
        messagebox.showinfo("Info", f"ファイル '{log_file_name}' は既に存在します")
    else:
        with open(log_file_name, 'w') as file:
            pass  # ファイルを空で作成
        messagebox.showinfo("Info", f"ファイル '{log_file_name}' を作成しました")
    ##if not os.path.exists(log_file_name):
        ##with open(log_file_name, "w"):
            #pass

    with open(log_file_path, "a") as log_file:    
    #print(nn)
     for j in range(nn):
        if Serial_Port !='':
            data=Serial_Port.readline()
            data=data.strip()
            data=data.decode('utf-8')
            st.insert(END, data + '\r\n')
            st.see(END)  # テキストを追加するたびにスクロールバーを下に移動
            st.update()
            log_file.write(data + '\n')
    print("Script is directly executed.")
    

def serial_open():
    global Serial_Port
    Serial_Port=serial.Serial()
    Serial_Port.port = serial_combo.get()
    Serial_Port.baudrate = serial_combo_2.get()
    Serial_Port.party = serial_combo_3.get()
    Serial_Port.timeout = 5
    Serial_Port.open()

def click_close():
    if messagebox.askokcancel("確認", "本当に閉じていいですか？"):
        ws.disconnect()
        ws2.disconnect()
        root.destroy()

if __name__ == '__main__':
    root = Tk()
    root.geometry("800x600")
    root.title('熱刺激装置コントロールソフト')

    st= scrolledtext.ScrolledText(
        root, 
        width=40, 
        height=15,
        font=("Helvetica",15)
    )
    st.pack()

    log_name_frame = ttk.Frame(root)
    log_name_frame.pack()

    log_name_label = ttk.Label(log_name_frame, text='ログ名：')
    log_name_label.pack(side=LEFT)

    log_name_entry = ttk.Entry(log_name_frame, width=20)
    log_name_entry.pack(side=LEFT)
    log_name_entry.insert(END, "")  # デフォルトのログファイル名を設定      

    
    # 刺激設定
    stim_flame = ttk.Frame(root)
    stim_flame.pack()
   
    stim_label = ttk.Label(stim_flame, text=' 刺激温度：')
    stim_label.pack(side=LEFT) 
    
    temp_list = [
    	'50', 
    	'49', '48', '47', '46', '45', '44', '43', '42', '41','40', 
    	'39', '38', '37', '36', '35', '34', '33', '32', '31','30',    
    	'29', '28',	'10', '0'  
    ]

    stim_combo = ttk.Combobox ( stim_flame , values = temp_list , width=3)
    stim_combo.set('42')
    stim_combo.pack(side=LEFT) 
    
    stim_label_2 = ttk.Label(stim_flame, text=' 刺激間隔：')
    stim_label_2.pack(side=LEFT) 
    
    interval_list = [ 
    	'30', '25', '20','15', '10', '5'    	    
    ]

    stim_combo_2 = ttk.Combobox ( stim_flame , values = interval_list , width=3)
    stim_combo_2.set('20')
    stim_combo_2.pack(side=LEFT) 
    
    stm = BooleanVar()
    stm.set( True )
    chk3 = ttk.Checkbutton(stim_flame, text=' 有効' , variable = stm)
    chk3.pack(side=LEFT)
    
    #serial設定
    serial_flame = ttk.Frame(root)
    serial_flame.pack()
    
    #portリストを取得
    serial_ports=[]
    for i,port in enumerate(serial.tools.list_ports.comports()):
        serial_ports.append(port.device)
    
    serial_label = ttk.Label(serial_flame, text=' Serial Setting：')
    serial_label.pack(side=LEFT)

    variable = StringVar ( ) 
    serial_combo = ttk.Combobox ( serial_flame , values = serial_ports , textvariable = variable , width=15)
    if not serial_ports==[]:
        serial_combo.set(serial_ports[0])
    serial_combo.pack(side=LEFT)
    
    serial_combo_2 = ttk.Combobox ( serial_flame , values = [9600, 115200] , width=7)
    serial_combo_2.set(9600)
    serial_combo_2.pack(side=LEFT)
    
    serial_combo_3 = ttk.Combobox ( serial_flame , values = ['N','O','E'] , width=2)
    serial_combo_3.set('N')
    serial_combo_3.pack(side=LEFT)
    
    #obs 1 save path name  
    try :
        host = os.getenv('OBS_HOST')
        port = os.getenv('OBS_PORT')
        password = os.getenv('OBS_PASS')
        ws = obsws(host, port, password)
        ws.connect()
        obspath = ws.call(requests.GetRecordDirectory()).getrecordDirectory()
        obsstate = 'enable'
    except :   
        obspath = '' 
        obsstate = 'disable'
        print('OBS 1 websocket is not connected')
        print(os.getenv('OBS_HOST'))
    
    obs1_flame = ttk.Frame(root)
    obs1_flame.pack()
    

    obs1_cam = BooleanVar()
    obs1_cam.set( True )
    obs1_check = ttk.Checkbutton(obs1_flame, text=' obs 1 path：',state=obsstate , variable = obs1_cam)
    obs1_check.pack(side=LEFT)

    obs1_label = ttk.Label(obs1_flame, text='OBS1保存先：')
    obs1_label.pack(side=LEFT)

 
    obs1_entry = ttk.Entry(obs1_flame, state=obsstate, width=30)
    obs1_entry.insert(END,obspath)
    obs1_entry.pack(side=LEFT)
    
    obs1b_flame = ttk.Frame(root)
    obs1b_flame.pack()
    obs1_video_name_label = ttk.Label(obs1b_flame, text='ファイル名：')
    obs1_video_name_entry = ttk.Entry(obs1b_flame)
    obs1b_flame.pack()
    obs1_video_name_label.pack(side=LEFT)
    obs1_video_name_entry.pack(side=LEFT)
    obs1_video_name_entry.insert(END,"")
    

    label = ttk.Label(root, text="フォルダ名を入力してください:")
    label.pack()

    entry = ttk.Entry(root, width=50)
    entry.pack()

    create_button = ttk.Button(root, text="フォルダを作成", command=create_folder)
    create_button.pack()



    #obs 2 save path name
    try :
        host = os.getenv('OBS_HOST_2')
        port = os.getenv('OBS_PORT_2')
        password = os.getenv('OBS_PASS_2')
        ws2 = obsws(host, port, password, timeout=5)
        ws2.connect()
        obspath2 = ws2.call(requests.GetRecordDirectory()).getrecordDirectory()
        obsstate2 = 'enable'
    except :    
        obspath2 = ''
        obsstate2 = 'disable'
        print('OBS 2 websocket is not connected')

    obs2_flame = ttk.Frame(root)
    obs2_flame.pack()

    obs2_cam = BooleanVar()
    obs2_cam.set( False )
    obs2_check = ttk.Checkbutton(obs2_flame, text=' obs 2 path：',state=obsstate2 , variable = obs2_cam)
    obs2_check.pack(side=LEFT)

    obs2_label = ttk.Label(obs2_flame, text='OBS2保存先：')
    obs2_label.pack(side=LEFT)

    
    obs2_entry = ttk.Entry(obs2_flame, state=obsstate2 , width=30)
    obs2_entry.insert(END,obspath2)
    obs2_entry.pack(side=LEFT)

    obs2_video_name_label = ttk.Label(obs2_flame, text='ファイル名：')
    obs2_video_name_label.pack(side=LEFT)

    obs2_video_name_entry = ttk.Entry(obs2_flame, width=20)
    obs2_video_name_entry.pack(side=LEFT)
    obs2_video_name_entry.insert(END, "video2")


    #ボタン
    update_button = Button(root, text="Update Filename", command=update_filename)
    update_button.pack()

    Button = ttk.Button(text=u'実験スタート', width=50)
    Button.bind("<Button-1>", trigger) 
    Button.pack()
   
    root.protocol("WM_DELETE_WINDOW", click_close)
    
    #ウィンドウの表示
    root.mainloop()

