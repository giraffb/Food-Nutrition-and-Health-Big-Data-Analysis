import torch  
from PIL import Image  
from torchvision import transforms  
# from uecfood import CustomResNet18
# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义转换函数，与训练时的转换保持一致  
# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import torch
from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models

# 定义自定义ResNet-50模型  
class ResNet50(nn.Module):  
    def __init__(self):  
        super(ResNet50, self).__init__()  
        self.resnet50 = torchvision.models.resnet50(pretrained=True)  # 加载预训练的ResNet-50模型  
        # 修改第一个卷积层以接受3通道输入（如有必要，更改输入通道数）  
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        # 更改最后一层以输出正确的类别数  
        num_ftrs = self.resnet50.fc.in_features  
        self.resnet50.fc = nn.Linear(num_ftrs, 100)  

    def forward(self, x):  
        return self.resnet50(x)  

# 加载预训练的ResNet-18模型  
model = ResNet50().to(device)  # 不载入预训练权重，以便加载本地权重  

# 加载本地的模型参数  
model.load_state_dict(torch.load('路径.pth'))  


# 加载图片并应用转换  
img_path = '路径/UECFOOD100/test/1/8_0.jpg'  # 替换为你的图片路径  
img = Image.open(img_path)  
img = transform(img)  # 转换图片  

# 检查是否有可用的GPU  
 
model.eval()  # 设置模型为评估模式  
# 测试图片并获取预测结果  
with torch.no_grad():  
    img = img.unsqueeze(0)  # 增加一个批次维度  
    img = img.to(device)  
    outputs = model(img)  
    
    # 获取预测类别及其概率  
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)  # 计算概率分布  
    class_idx = torch.argmax(probabilities, dim=1)[0]

# 将预测结果转换为类别名称  
class_dict = {1: "rice", 2: "eelsonrice", 3: "pilaf", 4: "chicken-'n'-eggonrice", 5: "porkcutletonrice",
                      6: "beefcurry", 7: "sushi", 8: "chickenrice", 9: "friedrice", 10: "tempurabowl", 11: "bibimbap",
                      12: "toast", 13: "croissant", 14: "rollbread", 15: "raisinbread", 16: "chipbutty",
                      17: "hamburger", 18: "pizza", 19: "sandwiches", 20: "udonnoodle", 21: "tempuraudon",
                      22: "sobanoodle", 23: "ramennoodle", 24: "beefnoodle", 25: "tensinnoodle", 26: "friednoodle",
                      27: "spaghetti", 28: "Japanese-stylepancake", 29: "takoyaki", 30: "gratin",
                      31: "sauteedvegetables", 32: "croquette", 33: "grilledeggplant",
                      34: "sauteedspinach", 35: "vegetabletempura", 36: "misosoup", 37: "potage", 38: "sausage",
                      39: "oden", 40: "omelet", 41: "ganmodoki", 42: "jiaozi", 43: "stew", 44: "teriyakigrilledfish",
                      45: "friedfish", 46: "grilledsalmon", 47: "salmonmeuniere", 48: "sashimi",
                      49: "grilledpacificsaury", 50: "sukiyaki", 51: "sweetandsourpork", 52: "lightlyroastedfish",
                      53: "steamedegghotchpotch", 54: "tempura", 55: "friedchicken", 56: "sirloincutlet",
                      57: "nanbanzuke", 58: "boiledfish", 59: "seasonedbeefwithpotatoes", 60: "hambargsteak",
                      61: "beefsteak", 62: "driedfish", 63: "gingerporksaute",
                      64: "spicychili-flavoredtofu", 65: "yakitori", 66: "cabbageroll", 67: "rolledomelet",
                      68: "eggsunny-sideup", 69: "fermentedsoybeans", 70: "coldtofu", 71: "eggroll",
                      72: "chillednoodle", 73: "stir-friedbeefandpeppers", 74: "simmeredpork",
                      75: "boiledchickenandvegetables", 76: "sashimibowl", 77: "sushibowl",
                      78: "fish-shapedpancakewithbeanjam", 79: "shrimpwithchillsource", 80: "roastchicken",
                      81: "steamedmeatdumpling", 82: "omeletwithfriedrice", 83: "cutletcurry", 84: "spaghettimeatsauce",
                      85: "friedshrimp", 86: "potatosalad", 87: "greensalad", 88: "macaronisalad",
                      89: "Japanesetofuandvegetablechowder", 90: "porkmisosoup", 91: "chinesesoup", 92: "beefbowl",
                      93: "kinpira-stylesauteedburdock", 94: "riceball", 95: "pizzatoast", 96: "dippingnoodles",
                      97: "hotdog", 98: "frenchfries", 99: "mixedrice", 100: "goyachanpuru"} 

predicted_name = class_dict.get(class_idx.item()+1)  
predicted_probability = probabilities[0][class_idx.item()].item()  # 获取对应类别的概率  

# 打印预测结果及其概率  
print(f'Predicted class index: {class_idx.item()+1}')  
print(f'Predicted class name: {predicted_name}')  
print(f'Predicted probability: {predicted_probability:.4f}')  # 打印概率，保留4位小数