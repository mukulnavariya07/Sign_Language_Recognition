import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from .config import CSV,MODEL
from .features import mirror_features

def train_pipeline(csv_path=CSV,model_path=MODEL,report_dir=None,cv_splits=5):
 csv_path=Path(csv_path);model_path=Path(model_path);out=Path(report_dir) if report_dir else model_path.parent.parent/'reports';out.mkdir(parents=True,exist_ok=True);model_path.parent.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(csv_path);cols=[c for c in df.columns if c.startswith('f')]
 if len(cols)!=63 or 'label' not in df:raise ValueError('Expected label and f0..f62 columns')
 X=df[cols].to_numpy(np.float32);y=df.label.astype(str).to_numpy();counts=pd.Series(y).value_counts()
 if len(counts)<2 or counts.min()<2:raise ValueError('Need >=2 classes and >=2 samples/class')
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
 Xtr=np.concatenate((Xtr,mirror_features(Xtr)));ytr=np.concatenate((ytr,ytr))
 folds=min(cv_splits,int(pd.Series(ytr).value_counts().min()));cv=StratifiedKFold(folds,shuffle=True,random_state=42)
 candidates={'RandomForest':RandomForestClassifier(n_estimators=180,class_weight='balanced',random_state=42,n_jobs=-1),
 'SVM':CalibratedClassifierCV(estimator=SVC(kernel='rbf',class_weight='balanced'),cv=3),
 'KNN':KNeighborsClassifier(n_neighbors=min(5,len(Xtr))),
 'MLP':MLPClassifier(hidden_layer_sizes=(96,48),max_iter=350,early_stopping=True,random_state=42)}
 scores={}
 for name,est in candidates.items():
  try:scores[name]=float(cross_val_score(est,Xtr,ytr,cv=cv,scoring='f1_macro').mean());print(name,'CV macro-F1',scores[name])
  except Exception as e:print('Candidate skipped',name,e)
 if not scores:raise RuntimeError('All model candidates failed CV')
 name=max(scores,key=scores.get);best=candidates[name].fit(Xtr,ytr);pred=best.predict(Xte)
 accuracy=float(accuracy_score(yte,pred));macro=float(f1_score(yte,pred,average='macro'));report=classification_report(yte,pred,zero_division=0)
 (out/'classification_report.txt').write_text(report,encoding='utf8')
 labels=sorted(set(yte)|set(pred));cm=confusion_matrix(yte,pred,labels=labels)
 fig,ax=plt.subplots(figsize=(max(8,.45*len(labels)),max(6,.4*len(labels))));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax);ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='Predicted',ylabel='Actual',title='ASL Confusion Matrix');plt.setp(ax.get_xticklabels(),rotation=45,ha='right');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=150);plt.close(fig)
 metrics={'best_model':name,'cv_macro_f1':scores,'test_accuracy':accuracy,'test_macro_f1':macro,'test_samples':len(yte),'class_counts':counts.to_dict(),'warning':'If data is synthetic, these are fake smoke-test results, not real recognition accuracy.'}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8');joblib.dump({'model':best,'feature_columns':cols,'labels':sorted(set(y)),'confidence_threshold':.65},model_path);print(json.dumps(metrics,indent=2));return metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,default=CSV);p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--report-dir',type=Path);p.add_argument('--cv',type=int,default=5);a=p.parse_args();train_pipeline(a.csv,a.model,a.report_dir,a.cv)
if __name__=='__main__':main()
