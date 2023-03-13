#%%

import yagmail
yag = yagmail.SMTP('icbad.updates99@gmail.com', 'wvliahqheajgzxzr')
contents = [
    f"{'hi'}"
]
yag.send('tjc119@ic.ac.uk', 'status', contents)



#%%

