import os
import zipfile
import shutil
from datetime import date


def ExportFilePath(_line):
    splitedLine=_line.split('/')
    path= splitedLine[0].replace('![','')
    return path

def UnZipFile():
    findZipFile=False
    
    for f in os.listdir():
        if f.endswith('.zip'):
            findZipFile=True
            print("unzip apks file")
            with zipfile.ZipFile(f) as notionZipFile:
                notionZipFile.extractall()
            os.remove(f)
            break
    return findZipFile

def FindMarkdownFile():
    notionMarkDownFile=''
    for f in os.listdir():
        if f.endswith('.md'):
            if f.lower().startswith('readme'):
                continue
            notionMarkDownFile=f
            break
    return notionMarkDownFile


def ModifiedMarkDownFile():

    #Input Information
    year=int(input("Year="))
    month=int(input("Month="))
    day=int(input("Day="))
    fileName=input("FileName=")
    subTitle=input("Subtitle=")
    cats=input("Categories=")
    tags=input("Tags=")
    

    currentTimeStr=date(year, month, day).isoformat()

    
    #Upzip file
    if UnZipFile()==False:
        print("Could not find ZipFile")
        return
    
    #Front Matter
    with open('customFrontMatter.txt','r') as f:
        customFrontMatter=f.read()

    
    #Read Notion Markdown
    notionMarkDownFile=FindMarkdownFile()
    notionMarkDownFolder=notionMarkDownFile.replace('.md','')

    titlesw=0
    fileName="{}-{}-{}".format(cats,tags,fileName)
    newfolderName="{}-{}".format(currentTimeStr,fileName)
    with open(notionMarkDownFile,'r') as f:
        
        n= f.read()
        lines=n.split('\n')
        path=''
        for line in lines:
            if line.startswith('# ') and titlesw == 0:
                path=line.replace('# ','')
                title=path
                n=n.replace(line,'')
                titlesw=1
            if line.startswith('!['):
                path=ExportFilePath(line)
                break
        n=n.replace(path,'/assets/img/post_img/{}'.format(newfolderName))
    
    frontMatter='---\nlayout: post\ntitle: "{}"\nsubtitle: "{}"\ncategories: {}\ntags: {}\n{}\n---\n'.format(title,subTitle,cats,tags,customFrontMatter)

    #Write Modified MarkDown
    newMarkdowntitle="_posts/{}.md".format(newfolderName)
    with open(newMarkdowntitle,'w') as f:
        f.write(frontMatter+n)
    
    #Remove md file
    os.remove(notionMarkDownFile)

    #Move Resouces file and Remove Folder
    print("SRC:"+os.curdir+'/{}'.format(notionMarkDownFolder))
    print("DES:"+os.curdir+'/assets/img/post_img/{}'.format(newfolderName))
    shutil.move(os.curdir+'/{}'.format(notionMarkDownFolder),os.curdir+'/assets/img/post_img/{}'.format(newfolderName))
   


if __name__ == '__main__':
    ModifiedMarkDownFile()
