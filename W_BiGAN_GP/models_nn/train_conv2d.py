 #%%imports
import os
import torch
from models_nn import model as model
from scripts import util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from __main__ import batch_size  ### import namespace of caller script

### def
def save_model(path_model, E,G,D,optimizer_EG,optimizer_D, dataloader, loss_all, collected_seqs, epoch, run_num):
    
    ## save model_para
    torch.save({
                'E_state_dict': E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_EG_state_dict': optimizer_EG.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                #'scheduler_D_state_dict': scheduler_D.state_dict(),  # if scheduler is being used in training
                #'scheduler_EG_state_dict': scheduler_EG.state_dict()
                }, path_model + 'modelsNoptimiser_state_dict_r'+ str(run_num)+'.tar')
    ### model
    torch.save(G, os.path.join(path_model, "G.pkl"))
    torch.save(E, os.path.join(path_model, "E.pkl"))
    torch.save(D, os.path.join(path_model, "D.pkl"))
    torch.save(dataloader, path_model+'dataloader.pt')

    # Additional information
    torch.save({
                'epoch': epoch+1,
                'loss_all': loss_all,
                "collected_seqs": collected_seqs
                }, path_model+'collectedseqs_loss_epochinfo_r'+ str(run_num)+'.json')
    return

def load_model(path_model,E, G, D, optimizer_EG, optimizer_D, run_num):
    ## save model_para
    stat = torch.load(path_model + 'modelsNoptimiser_state_dict_r'+ str(run_num-1)+'.tar')
    dataloader = torch.load(path_model +'dataloader.pt')

    stat_list = {'E_state_dict': E,
    'G_state_dict': G,
    'D_state_dict': D,
    'optimizer_EG_state_dict': optimizer_EG,
    'optimizer_D_state_dict': optimizer_D
    }
    for i, (key ,value) in enumerate(stat_list.items()):
        if i < 3:
            value.to(device)
        value.load_state_dict(stat[key])
        
    #JSON object as a dictionary
    f = torch.load(path_model + 'collectedseqs_loss_epochinfo_r'+ str(run_num-1)+'.json')
    loss_all = f["loss_all"]
    epoch = f["epoch"]
    return E,G,D,optimizer_EG,optimizer_D,dataloader, loss_all, epoch



def training(path_model, epoch, train_data=None, batch_size= 128, run_num = None):
    
    ### varriables
    G, E, D = model.get_model_and_optimizer() 
    E = E.to(device)
    G = G.to(device)
    D = D.to(device)
    E.apply(model.weights_init)
    G.apply(model.weights_init)
    D.apply(model.init_weights)
    collected_seqs = {}
    table = util.get_conversion_table("/content/bigan-pep-test/data_master/physical_chemical_6.txt")
    optimizer_EG = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), 
                                    lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(D.parameters(), 
                                lr=2e-5, betas=(0.5, 0.999), weight_decay=1e-5)
    if run_num == 1:
        L_D,L_EG,L_E,L_G = [], [],[],[]
        epoch_i = 0               
        #optimizers with weight decay
        dataloader = train_data
    else:
        all_parms = load_model(path_model, E, G, D, optimizer_EG, optimizer_D, run_num)
        E,G,D,optimizer_EG,optimizer_D,dataloader, loss_all, epoch_i = all_parms
        [L_D,L_EG,L_E,L_G] = loss_all
    epoch_f = epoch_i + epoch
    EG_update,D_update = False,True
    D_Iter = 5
    EG_Iter = 1
    d_iter = 0
    eg_iter = 0
    ### training loop
    for epoch in range(epoch_i, epoch_f):
        D_loss_acc,EG_loss_acc,E_loss_acc,G_loss_acc = 0., 0., 0., 0.
        D.train()
        E.train()
        G.train()
        
        for i, data in enumerate(dataloader):
            real_seqs = data[0]
            real_seqs = real_seqs.to(device)
            #initialize z
            z = torch.randn(128,100, 1, 1)
            z = z.to(device)
           
            #compute G(z) and E(X)
            Gz = G(z)
            EX = E(real_seqs)
            DG=D(Gz,z)
            DE=D(real_seqs,EX)
            #compute losses and append in list
            loss_instance = model.loss_GP(D,real_seqs,EX,Gz,z)
            loss_D, loss_EG, loss_E, loss_G = loss_instance.loss()
            
            #Discriminator training
            if D_update:
                optimizer_D.zero_grad()
                D_loss_acc += loss_D.item()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                d_iter = d_iter+1

                if d_iter == D_Iter:
                    d_iter = 0
                    EG_update,D_update = True,False
                    continue


            #Encoder & Generator training
            if EG_update:
                optimizer_EG.zero_grad()
            # loss_EG = model.EG_loss(DG, DE)
                EG_loss_acc += loss_EG.item()
                loss_EG.backward()
                optimizer_EG.step()
                eg_iter =eg_iter + 1

                if eg_iter == EG_Iter:
                    eg_iter = 0
                    EG_update,D_update = False,True
                    continue
    
        # print(losses) and append losses
        if (epoch + 1) % 10 == 0 and i == len(dataloader) - 1:
            print('Epoch [{}/{}], Avg_Loss_D: {:.4f}, Avg_Loss_EG: {:.4f}'
                  .format(epoch + 1, epoch_f , D_loss_acc / i, EG_loss_acc / i))
            
            L_D.append(D_loss_acc / i)
            L_EG.append(EG_loss_acc / i)
            L_E.append(E_loss_acc / i)
            L_G.append(G_loss_acc / i)
        # generate and collect seqs   
        if (epoch +1) % 100 == 0 and i == len(dataloader) - 1:
            with torch.no_grad():
                D.eval()
                E.eval()
                G.eval()
                G = G.cpu()
                z = z.cpu()
                generated_seqs = util.generate_seqs(G, table, z, epoch)
                G=G.cuda()
                collected_seqs[epoch] = generated_seqs
                util.write_fasta(generated_seqs, os.path.join(path_model, "final_generated_seq.fasta"))
                D.train()
                E.train()
                G.train()
    
    loss_all = [L_D,L_EG,L_E,L_G]
    save_model(path_model, E,G,D,optimizer_EG,optimizer_D, dataloader, loss_all, collected_seqs, epoch, run_num)
    return G, loss_all, collected_seqs
# from torchsummary import summary
# def models_summary(path_model):
#     G = model.Generator(100,6,64)
#     E = model.Encoder(6,64)
#     D = model.Discriminator()
    
#     G = G.to(device)
#     E = E.to(device)
#     D = D.to(device)
#     result_g = summary(G, input_size = (100, 1, 1), batch_size = batch_size)
#     result_e = summary(E, input_size = (1, 30, 6), batch_size = batch_size)
#     result_d = summary(D, [(1,30,6 ), (100, 1, 1)])
    
#     with open(path_model + 'models_summary.txt','w') as f:
#         f.write("\n\n ####### Encoder ####### \n\n")
#         print(result_e)
#         f.write(result_e)
#         f.write("\n\n ####### Generator ####### \n\n")
#         f.write(result_g)
#         f.write("\n\n ####### Discriminator ####### \n\n")
#         f.write(result_d)
#     return



