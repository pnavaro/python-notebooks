from operator import itemgetter
from dataclasses import dataclass

@dataclass
class GroceryItem:
    """ Grocery list item """

    name : str
    price_without_cut : float
    cut_percentage : int
    category : str
    vat_percentage : int
    quantity : int
    ingredients : list

    def get_total_price(self):
        res = self.price_without_cut * (1 + self.vat_percentage / 100) * (1 - self.cut_percentage / 100) * self.quantity
        return round(res, 2)

    def get_total_vat(self):
        res = self.price_without_cut * self.vat_percentage / 100 * (1 - self.cut_percentage / 100) * self.quantity
        return round(res, 2)

    def get_total_cut(self):
        res = self.price_without_cut * (1 + self.vat_percentage / 100) * self.cut_percentage / 100 * self.quantity
        return round(res, 2)

    def __repr__(self):
        return f"{self.name:s} x {self.quantity:d}"


class GroceryList:

    def __init__(self, *args):

        self.items = args

    def items_with_meat(self):
        return [a for a in self.items if a.category == "Meat"]

    def prices_with_vat(self):
        res = {}
        for a in self.items:
            res[a.name] = a.get_total_price()
        return res

    def ingredients_list(self):
        res = []
        for a in self.items:
            res += a.ingredients
        return set(res)

    def total_invoice(self):
        return sum([a.get_total_price() for a in self.items])

    def total_for(self, category):
        return round(sum([a.get_total_price() for a in self.items if a.category == category]), 2)

    def price_by_category(self):
        res = {}
        for a in self.items:
            try:
                res[a.category] += a.get_total_price()
            except KeyError:
                res[a.category] = a.get_total_price()
        return res

    def total_vat(self):
        return round(sum([a.get_total_vat() for a in self.items]), 2)

    def total_cut(self):
        return round(sum([a.get_total_cut() for a in self.items]), 2)

    def top_ingredients(self, n):
        res = {}
        for a in self.items:
            for i in a.ingredients:
                try:
                    res[i] += 1
                except KeyError:
                    res[i] = 1
        return sorted(res.items(), key=itemgetter(1), reverse=True)[:n]

    def all_item_names(self):
        return [a.name for a in self.items]

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        yield from self.items


if __name__ == "__main__":

    beef = GroceryItem("Beef", 12.3, 0, "Meat", 10, 2, ["Beef"])
    print(beef)
    print(f"Total cut   : {beef.get_total_cut():.2f} \u20ac ")
    print(f"Total price : {beef.get_total_price():.2f} \u20ac ")
    print(f"Total VAT   : {beef.get_total_vat():.2f} \u20ac ")

    shopping_list = GroceryList(
        GroceryItem("Beef", 12.3, 0, "Meat", 10, 2, ["Beef"]),
        GroceryItem("Pork", 7.99, 5, "Meat", 10, 1, ["Pork"]),
        GroceryItem("Tomato Sauce", 2, 0, "Can", 10, 3, ["Tomato", "Water", "Salt", "Sugar", "Preservatives"]),
        GroceryItem("Beans", 3.5, 10, "Can", 10, 5, ["Beans", "Water", "Salt", "Preservatives"]),
        GroceryItem("Tuna", 1.50, 0, "Can", 20, 4, ["Fish", "Oil", "Salt", "Water", "Preservatives"])
    )

    print(shopping_list.items)
    print(f"Number of Grocery Items : {len(shopping_list)}")
    print(shopping_list[::-1])
    for item in shopping_list:
        print(f"{item} = {item.get_total_price():7.2f} \u20ac)")

    print(f"Articles with meat are : {shopping_list.items_with_meat()}")
    print(f"Full prices are : {shopping_list.prices_with_vat()}")
    print(f"Ingredients : {shopping_list.ingredients_list()}")
    print(f"Total  : {shopping_list.total_invoice()}")
    print(f"Total for meat category : {shopping_list.total_for('Meat')}")
    print(f"Prices by category : {shopping_list.price_by_category()}")
    print(f"VAT amount : {shopping_list.total_vat()}")
    print(f"Cut value : {shopping_list.total_cut()}")
    print(f"First three ingedients : {shopping_list.top_ingredients(3)}")
    print(f"All articles names : {shopping_list.all_item_names()}")
