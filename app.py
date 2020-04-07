import argparse, logging, shelve, os, cv2, shutil
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s  : %(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

shelfFile = shelve.open('data')
classifier_path = 'facerecogniser.pkl'
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

parser = argparse.ArgumentParser(description="Facial Recognition System")
parser.add_argument('-a', '--add-face', metavar='name', help='name of the person to recognise')
parser.add_argument('-c', '--configure', metavar='path', help='path to store cascade')
parser.add_argument('-d', '--delete-face', metavar='name', help='name of the person to delete')
parser.add_argument('-l', '--list', action='store_true', help='list out the faces the app recognises')
parser.add_argument('-r', '--recognise', action='store_true', help='recognise the face')
parser.add_argument('-x', '--reset', action='store_true', help='reset the application')
args = parser.parse_args()

logger.info(args)

def preprocess(img, size=(64, 64)):
    if img.shape < size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.equalizeHist(img)
    return img
    
def prepareDataSet():
    X = []
    y = []

    for folderName, subfolders, filenames in os.walk(shelfFile['path']):
        for filename in filenames:
            X.append(cv2.imread(os.path.join(folderName,filename), 0))
            y.append(folderName.split('\\')[-1])
    X.pop(0)
    y.pop(0)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y    

# Configure
if args.configure is not None:
    logger.info('Configuring')
    if not os.path.exists(str(args.configure)):
        os.mkdir(str(args.configure))
    shelfFile['path'] = str(args.configure)
    logger.debug(f'Contents of shelve: {list(shelfFile.items())}')
    logger.info('Configured')

# Add face
elif args.add_face is not None:
    if 'faces' not in shelfFile.keys():
        shelfFile['faces'] = []
    
    if args.add_face in shelfFile['faces']:
        logger.info('face already exists')
    else:
        os.mkdir(f'{shelfFile["path"]}/{args.add_face}')
        logger.info('Adding Face')
        logger.info('Starting to capture images')
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = haarcascade.detectMultiScale(gray) 
        
            for (x,y,w,h) in rects: 
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.imshow('frame', frame)
            
            if len(rects) !=0:
                (x,y,w,h) = rects[0]
                count += 1
                img = preprocess(gray[y:y+w, x:x+h])
                cv2.imwrite(f'{shelfFile["path"]}/{args.add_face}/{count}.png', img)
                logger.debug(f'saved{count}')
                
            if cv2.waitKey(1) & count == 250:
                break

        cap.release()
        cv2.destroyAllWindows()
        logger.info('images captured')
        
        logger.info('training started')
        # images = []
        # labels = []
        # for folderName, subfolders, filenames in os.walk(shelfFile['path']):
        #     for filename in filenames:
        #         images.append(cv2.imread(folderName + '/' + filename, 0))
        #         labels.append(len(shelfFile['faces']))

        # shelfFile['faces'] = shelfFile['faces']+[str(args.add_face)]
        logger.debug(shelfFile['faces'])
        logger.info('training completed')
    
# Delete    
elif args.delete_face is not None:
    if args.delte_face in shelfFile['faces']:
        shelfFile['faces'].remove(args.delte_face)
        shutil.rmtree(shelfFile['path']+'/'+args.delete_face)
    # TODO: Retrain    
    
# List
elif args.list == True:
    logger.info('Face List')
    print(shelfFile['faces'])

# Recognise
elif args.recognise == True:
    logger.info('Recognising Face')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = haarcascade.detectMultiScale(gray) 

        
        if len(rects) !=0:
            (x,y,w,h) = rects[0]   
            img = preprocess(gray[y:y+w, x:x+h])
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2) 
            
            # TODO Predictions
            
            
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Reset
elif args.reset == True:
    logger.info('reseting application')
    shelfFile['faces'] = []
    shutil.rmtree(shelfFile['path'])
    shelfFile['path'] = ''
    logger.info('application reset')
    
else:
    logger.info('Invalid Arguments')

logger.debug(f'ShelfFile {list(shelfFile.items())}')
shelfFile.close()
logger.info('Application is exiting')