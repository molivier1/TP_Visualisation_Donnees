def clients_shape(clients):
    return clients.shape
def clients_type(clients):
    return clients.info()
def clients_valeur_manquantes(clients):
    return clients[clients.isnull().sum(axis=1) > 0].copy()
def clients_duplicated_values(clients):
    return clients[clients.duplicated()]