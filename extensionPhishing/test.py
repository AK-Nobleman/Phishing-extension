import torch

example = "https://docs.google.com/document/d/15mQpyABNvt5_FYwXiwSB4956GO5igcoDWDMLg3p2i2E/edit"

urllen = len(example)
dots = example.count(".")
hypens = example.count("-")
underline = example.count("_")
slash = example.count("/")
question = example.count("?")
equal = example.count("=")
at = example.count("@")
ands = example.count("&")
exclamation = example.count("!")
space = example.count(" ")
tilde = example.count("~")
comma = example.count(",")
plus = example.count("+")
asterisk = example.count("*")
hashtag = example.count("#")
dollar = example.count("$")
percent = example.count("%")


http=example.count("http")
if(http>1):
    http-=1
redirecting = http + example.count("url=") + example.count("redirect") + example.count("next=") + example.count("out=") + example.count("view")

elements = [urllen, dots, hypens, underline, slash, question, equal, at, ands, exclamation, space, tilde, comma, plus, asterisk, hashtag, dollar,percent, redirecting]

testing = torch.tensor(elements)
print(testing)