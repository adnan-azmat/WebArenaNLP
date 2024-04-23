import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def main():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
        trust_remote_code=True,
    )

    ft_model = PeftModel.from_pretrained(base_model, "mistralai-webdata-finetune/checkpoint-2300")
    # ft_model.push_to_hub("daparasyte/mistral-webagent")

    # eval_prompt = """From the MedQuad MedicalQA Dataset: Given the following medical question and question type, provide an accurate answer:

    # ### Question type:
    # symptoms

    # ### Question:
    # What are the symptoms of Norrie disease ?

    # ### Answer:
    # """

    eval_prompt = """You are an expert web agent. Given that the user's objective and information about the last page visited, with observation, provide an accurate action to choose next.

    ### Information:
    We need to find 'What is the top-1 best-selling product in 2022' from this site
    URL: http://metis.lti.cs.cmu.edu:7780/admin/admin/dashboard/

    Based on the current observed state, suggest what action to do next.

    ### observation:
    Tab 0 (current): Dashboard / Magento Admin

    [1] RootWebArea 'Dashboard / Magento Admin' focused: True
        [1306] link 'Magento Admin Panel'
            [1307] img 'Magento Admin Panel'
        [1218] menubar '' orientation: horizontal
            [1220] link '\ue604 DASHBOARD'
            [1223] link '\ue60b SALES'
            [1229] link '\ue608 CATALOG'
            [1235] link '\ue603 CUSTOMERS'
            [1241] link '\ue609 MARKETING'
            [1247] link '\ue602 CONTENT'
            [1253] link '\ue60a REPORTS'
            [1271] link '\ue60d STORES'
            [1277] link '\ue610 SYSTEM'
            [1283] link '\ue612 FIND PARTNERS & EXTENSIONS'
        [83] heading 'Dashboard'
        [84] link '\ue600 admin'
        [86] link '\ue607'
        [1862] textbox '\ue60c' required: False
        [51] main ''
            [89] StaticText 'Scope:'
            [153] button 'All Store Views' hasPopup: menu
            [156] link '\ue633 What is this?'
            [291] button 'Reload Data'
            [292] HeaderAsNonLandmark ''
                [350] StaticText 'Advanced Reporting'
            [294] link 'Go to Advanced Reporting \ue644'
            [442] StaticText 'Chart is disabled. To enable the chart, click '
            [443] link 'here'
            [660] StaticText 'Revenue'
            [531] StaticText '$0.00'
            [661] StaticText 'Tax'
            [662] StaticText 'Shipping'
            [663] StaticText 'Quantity'
            [545] StaticText '0'
            [1119] tablist '' multiselectable: False orientation: horizontal
                [1121] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers' expanded: True selected: True controls: grid_tab_ordered_products_content
                    [1129] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                [1123] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products' expanded: False selected: False controls: ui-id-1
                    [1131] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products'
                [1125] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers' expanded: False selected: False controls: ui-id-2
                    [1133] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers'
                [1127] tab 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers' expanded: False selected: False controls: ui-id-3
                    [1135] link 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers'
            [1140] tabpanel 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                [1148] table ''
                    [1171] row ''
                        [1177] columnheader 'Product' required: False
                        [1178] columnheader 'Price' required: False
                        [1179] columnheader 'Quantity' required: False
                    [1172] row 'http://metis.lti.cs.cmu.edu:7780/admin/catalog/product/edit/id/29/'
                        [1180] gridcell 'Sprite Stasis Ball 65 cm' required: False
                        [1181] gridcell '$27.00' required: False
                        [1182] gridcell '6' required: False
            [450] StaticText 'Lifetime Sales'
            [453] StaticText '$0.00'
            [455] StaticText 'Average Order'
            [460] StaticText 'Last Orders'
            [461] table ''
                [672] row ''
                    [776] columnheader 'Customer' required: False
                    [777] columnheader 'Items' required: False
                    [778] columnheader 'Total' required: False
                [673] row 'http://metis.lti.cs.cmu.edu:7780/admin/sales/order/view/order_id/299/'
                    [779] gridcell 'Sarah Miller' required: False
                    [780] gridcell '5' required: False
                    [781] gridcell '$194.40' required: False
                [674] row 'http://metis.lti.cs.cmu.edu:7780/admin/sales/order/view/order_id/65/'
                    [782] gridcell 'Grace Nguyen' required: False
                    [783] gridcell '4' required: False
                    [784] gridcell '$190.00' required: False

    ### Action:
    """

    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        print(f'Tuned Model output:\n\n{eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=64)[0], skip_special_tokens=True)}')


if __name__=='__main__':
    main()